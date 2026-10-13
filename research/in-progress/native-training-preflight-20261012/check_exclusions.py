"""Synthetic exclusion audit of the retained candidate training runtime.

No trained model, optimizer updates, reference payloads or network calls.
This is not a reproduction of the complete Reader training environment.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    source=a.repo/'research/lab/reports/fortysixth-experiment/runtime-source'
    sys.path.insert(0,str(source))
    from vesuvius.ink_detection.config import InkDataConfig,InkConfig
    from vesuvius.ink_detection.data.dataset import InkDataset
    from vesuvius.ink_detection.data.patch_finding_default import find_segment_patches
    from vesuvius.ink_detection.types import Segment
    from vesuvius.ink_detection.training.train import prepare_loss_inputs
    from vesuvius.ink_detection.training.deep_supervision import concatenate_deep_supervision_ignore
    from vesuvius.ink_detection.training.losses import create_loss
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    authored=dict(mode='flat',model_type='vesuvius_unet_3d_stem_2d',in_channels=1,patch_size=[17,128,128],
        patch_overlap=.25,patch_min_labeled_coverage=0.,
        image_normalization=dict(mode='robust_mad',percentile_lower=1.,percentile_upper=99.),
        flat_z_window_jitter=dict(enabled=True,window_depth=17,max_offset=2,probability=1.,padding='forbidden'),
        datasets=[dict(segments_path='synthetic',volume_scale=0)],
        targets=dict(ink=dict(activation='none',out_channels=1,z_projection_mode='none')),
        loss=dict(bce_label_smoothing=.5,dice_label_smoothing=0.),out_dir='unused',dataloader_workers=1)
    data=InkDataConfig.from_mapping(authored);criterion=create_loss(InkConfig.from_mapping(authored))
    seg=Segment(data,data.datasets[0],0,'synthetic',Path('synthetic'),'synthetic','image',
                Path('labels'),Path('supervision'),Path('validation'))
    rng=np.random.default_rng(120139);shape=(21,192,192)
    image=rng.integers(1,256,shape,dtype=np.uint8)
    sup=np.zeros(shape,np.uint8);sup[:,:144,:144]=255
    val=np.zeros(shape,np.uint8);val[:,64:144,64:144]=255
    labels=np.zeros(shape,np.uint8);labels[:,16:32,16:32]=255
    allowed=(sup>0)&(val==0)
    changed=labels.copy();changed[~allowed]=255-changed[~allowed]
    arrays=dict(image=image,labels=labels,supervision=sup,validation=val)
    counter=dict(arrays,labels=changed)
    def discover(segment,volumes):
        return find_segment_patches(segment,lambda path,scale:volumes[str(path)])
    train_a,valid_a=discover(seg,arrays);train_b,valid_b=discover(seg,counter)
    assert [x.bbox for x in train_a]==[x.bbox for x in train_b]
    assert [x.bbox for x in valid_a]==[x.bbox for x in valid_b]
    def loader(volumes,patches):
        ds=InkDataset(data,do_augmentations=False,patches=patches,segments=[seg])
        ds._open=lambda path,scale:volumes[str(path)]
        return ds
    la,lb=loader(arrays,train_a),loader(counter,train_b)
    loss_records=[];raw_target_differences=0
    def compute(sample):
        batch={k:v.unsqueeze(0) for k,v in sample.items()}
        weight=torch.tensor(.4,requires_grad=True);bias=torch.tensor(-.3,requires_grad=True)
        predictions=batch['image'].mean(2)*weight+bias
        pred,target,ignore=prepare_loss_inputs(predictions,batch,mode='flat')
        joined=concatenate_deep_supervision_ignore(target,ignore,pred)
        loss=criterion(pred,joined);loss.backward()
        return dict(loss=float(loss.detach()),gradient=np.array([weight.grad.item(),bias.grad.item()]),
                    target=target.detach(),ignore=ignore.detach(),image=batch['image'])
    for i in range(len(la)):
        l=compute(la[i]);rr=compute(lb[i])
        assert torch.equal(l['image'],rr['image']) and torch.equal(l['ignore'],rr['ignore'])
        valid=l['ignore']==0
        assert torch.equal(l['target'][valid],rr['target'][valid])
        raw_target_differences+=int((l['target']!=rr['target']).sum())
        assert l['loss']==rr['loss'] and np.array_equal(l['gradient'],rr['gradient'])
        loss_records.append(dict(bbox=list(train_a[i].bbox),loss=l['loss'],gradient=l['gradient'].tolist(),
                                 input_sha256=digest(l['image'].numpy())))
    # Negative control: a nonzero label-coverage threshold reads unmasked labels.
    filtering=dataclasses.replace(data,patch_finding=dataclasses.replace(data.patch_finding,min_labeled_coverage=.1))
    filtered_seg=dataclasses.replace(seg,data_config=filtering)
    fa,_=discover(filtered_seg,arrays);fb,_=discover(filtered_seg,counter)
    introduced=sorted(set(p.bbox for p in fb)-set(p.bbox for p in fa))
    assert introduced, 'Negative control failed to detect label-dependent discovery'
    # Boundary of the flat recipe: depth-varying supervision is not a repeated 2D label.
    sample=la[0];sample={k:v.clone() for k,v in sample.items()}
    sample['supervision_mask'].zero_();sample['supervision_mask'][:,8,20:24,20:24]=255
    sample['inklabels'].zero_();different={k:v.clone() for k,v in sample.items()}
    different['inklabels'][:,0,20:24,20:24]=255
    l=compute(sample);rr=compute(different)
    effective_changes=int(((l['target']!=rr['target'])&(l['ignore']==0)).sum())
    assert effective_changes==16 and not np.array_equal(l['gradient'],rr['gradient'])
    names=['config.py','data/dataset.py','data/patch_finding_default.py','data/normalization.py',
           'training/train.py','training/deep_supervision.py','training/losses.py']
    pins=[]
    for name in names:
        f=source/'vesuvius/ink_detection'/name
        pins.append(dict(path=str(f.relative_to(a.repo)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    result=dict(status='COMPLETED_WITH_TWO_CONFIGURATION_BOUNDARIES',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_pins=pins,
        tested_contract='Flat repeated-2D masks and labels, min_labeled_coverage=0, augmentation disabled, retained runtime',
        positive_control=dict(status='PASS',training_patches=len(train_a),validation_patches=len(valid_a),
            changed_excluded_voxels=int((labels!=changed).sum()),membership_unchanged=True,
            inputs_effective_targets_loss_and_gradients_exact=True,
            ignored_target_value_differences=raw_target_differences,per_patch=loss_records),
        coverage_negative_control=dict(min_labeled_coverage=.1,original_training_patches=len(fa),
            perturbed_training_patches=len(fb),introduced_patch_bboxes=[list(x) for x in introduced],
            conclusion='Nonzero label-coverage filtering can depend on excluded label values; release-like zero threshold passes'),
        depth_negative_control=dict(changed_supervised_2d_targets=effective_changes,
            original_loss=l['loss'],perturbed_loss=rr['loss'],original_gradient=l['gradient'].tolist(),perturbed_gradient=rr['gradient'].tolist(),
            conclusion='Flat amax projection can import excluded labels from another depth plane if supervision varies with depth'),
        limits=['No real training dataset or augmentation/jitter pipeline qualified',
                'No claim that released Reader training encountered either negative-control condition',
                'Frozen model comparison unchanged; no trained-model forward or optimizer update'],
        trained_model_forwards=0,optimizer_updates=0,paid_compute=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_pins','positive_control']},indent=2))


if __name__=='__main__':main()
