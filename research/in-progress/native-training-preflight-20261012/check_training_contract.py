"""Qualify default augmentation/jitter and a training-only label canonicalizer.

Synthetic examples only. Does not modify any frozen runtime or live study.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    source=a.repo/'research/lab/reports/fortysixth-experiment/runtime-source'
    sys.path.insert(0,str(source))
    from vesuvius.ink_detection.config import InkDataConfig,InkConfig
    from vesuvius.ink_detection.types import Segment,Patch
    from vesuvius.ink_detection.data.dataset import InkDataset
    import vesuvius.ink_detection.data.dataset as ds_module
    from vesuvius.ink_detection.training.train import prepare_loss_inputs
    from vesuvius.ink_detection.training.deep_supervision import concatenate_deep_supervision_ignore
    from vesuvius.ink_detection.training.losses import create_loss
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    authored=dict(mode='flat',model_type='vesuvius_unet_3d_stem_2d',in_channels=1,patch_size=[17,128,128],
        patch_overlap=.25,patch_min_labeled_coverage=0.,
        image_normalization=dict(mode='robust_mad',percentile_lower=1.,percentile_upper=99.),
        flat_z_window_jitter=dict(enabled=True,window_depth=17,max_offset=2,probability=1.,padding='forbidden'),
        augmentation_preset='default',augmentation_rotation_axes=[0],
        datasets=[dict(segments_path='synthetic',volume_scale=0)],
        targets=dict(ink=dict(activation='none',out_channels=1,z_projection_mode='none')),
        loss=dict(bce_label_smoothing=.5,dice_label_smoothing=0.),out_dir='unused',dataloader_workers=1)
    config=InkDataConfig.from_mapping(authored);criterion=create_loss(InkConfig.from_mapping(authored))
    seg=Segment(config,config.datasets[0],0,'synthetic',Path('synthetic'),'synthetic','image',
                Path('labels'),Path('supervision'),Path('validation'))
    patch=Patch(seg,(2,0,0,19,128,128))
    rng=np.random.default_rng(120139);image=rng.integers(1,256,(21,128,128),dtype=np.uint8)
    support=np.zeros((128,128),np.uint8);support[8:120,8:112]=255
    validation=np.zeros_like(support);validation[64:120,64:112]=255
    ink=np.zeros_like(support);ink[24:48,16:56]=255
    offsets=[];original=ds_module.flat_z_window_bbox
    def traced(*args,**kwargs):
        bbox,offset=original(*args,**kwargs);offsets.append(offset);return bbox,offset
    ds_module.flat_z_window_bbox=traced
    def dataset(labels,sup,val):
        arrays=dict(image=image,labels=labels,supervision=sup,validation=val)
        ds=InkDataset(config,do_augmentations=True,patches=[patch],segments=[seg])
        ds._open=lambda path,scale:arrays[str(path)]
        return ds
    def outcome(ds,seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        sample=ds[0];batch={k:v.unsqueeze(0) for k,v in sample.items()}
        params=torch.tensor([.4,-.3],requires_grad=True)
        logits=batch['image'].mean(2)*params[0]+params[1]
        pred,target,ignore=prepare_loss_inputs(logits,batch,mode='flat')
        loss=criterion(pred,concatenate_deep_supervision_ignore(target,ignore,pred));loss.backward()
        return batch['image'],target,ignore,loss.detach(),params.grad
    def equal(l,rr):
        assert torch.equal(l[0],rr[0]) and torch.equal(l[2],rr[2])
        mask=l[2]==0;assert torch.equal(l[1][mask],rr[1][mask])
        assert torch.equal(l[3],rr[3]) and torch.equal(l[4],rr[4])
    rows=[]
    for representation in ['repeated_2d','central_plane_only']:
        sup=np.zeros_like(image);val=np.zeros_like(image);labels=np.zeros_like(image)
        planes=list(range(21)) if representation=='repeated_2d' else [10]
        for z in planes:sup[z]=support;val[z]=validation;labels[z]=ink
        active=(sup>0)&(val==0)
        alt=labels.copy()
        for z in planes:
            mask=~active[z];alt[z,mask]=255-alt[z,mask]
        all_excluded=labels.copy();all_excluded[~active]=255-all_excluded[~active]
        # This is a new training-only view, not an edit to reference data.
        canonical=np.where(active,all_excluded,0).astype(np.uint8)
        base=dataset(labels,sup,val);counter=dataset(alt,sup,val);guarded=dataset(canonical,sup,val)
        for seed in range(24):
            first=outcome(base,seed);equal(first,outcome(counter,seed));equal(first,outcome(guarded,seed))
            rows.append(dict(representation=representation,seed=seed,loss=float(first[3]),gradient=first[4].tolist(),
                             image_sha256=hashlib.sha256(first[0].numpy().tobytes()).hexdigest()))
    assert sorted(set(offsets))==[-2,-1,0,1,2]
    packages={x.metadata['Name']:x.version for x in importlib.metadata.distributions()}
    result=dict(status='PASS',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        paired_seed_representation_cases=48,comparisons=96,observed_native_depth_offsets=sorted(set(offsets)),
        default_augmentation_enabled=True,input_target_loss_gradient_equality=True,
        canonicalizer='training_labels = where((supervision > 0) AND (validation == 0), stored_labels, 0), before augmentation/projection',
        canonicalizer_scope='Training view only; never overwrite the retained source or validation-reference labels',
        central_plane_evidence='Pinned Reader v2 build_labels writes only the central plane; other planes are canonical fill zero',
        scope='Synthetic loader/loss and two-parameter arithmetic probe; no trained-model or real-source/registration qualification',
        cases=rows,packages=packages,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trained_model_forwards=0,optimizer_updates=0,paid_compute=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','packages']},indent=2))


if __name__=='__main__':main()
