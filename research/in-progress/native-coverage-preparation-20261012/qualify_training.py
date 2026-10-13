"""Bounded real-input backward/accumulation/resume check, not a learning study."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import zarr
from matched_training import (slot, stream_totals, partial_stream_loss,
                             require_sample_independent_model, tensor_state_hash)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(a):
    a.work.mkdir(exist_ok=True)
    if (a.work/'TRAINING_GO.json').exists():raise FileExistsError('Already frozen')
    source=a.source
    rows=json.loads((source/'RESULTS.json').read_text())['sites']
    chosen=[]
    for w in ['w035','w039']:
        rr=next(s['rows'] for s in rows if s['winding']==w)
        for label,fn in [('negative',lambda r:r['ink_supervised']==0),
                         ('two_class',lambda r:0<r['ink_supervised']<r['supervised_pixels'])]:
            row=sorted([r for r in rr if r['eligible'] and fn(r)],key=lambda r:r['core_id'])[0]
            chosen.append(dict(row,winding=w,fixture_role=label))
    chunk_rows=json.loads((source/'ct/REQUEST_PLAN.json').read_text())['requests']
    source_pins={};raw=[];targets=[];masks=[]
    for row in chosen:
        w=row['winding'];x,y=row['x'],row['y']
        cr=next(r for r in chunk_rows if r['winding']==w and r['chunk_zyx']==[0,y//128,x//128])
        cp=source/'ct'/cr['relative_path'];receipt=json.loads((source/'ct/receipts'/f'{cr["id"]}.json').read_text())
        assert sha(cp)==receipt['sha256'] and cp.stat().st_size==receipt['bytes']==28*128*128
        source_pins[str(cp)]={'sha256':sha(cp),'bytes':cp.stat().st_size}
        raw.append(np.fromfile(cp,np.uint8).reshape(28,128,128)[4:25].copy())
        for role in ['inklabels','supervision']:
            for receipt in json.loads((source/'REQUEST_LEDGER.json').read_text())['requests']:
                if receipt['winding']==w and receipt['role']==role:
                    path=source/receipt['path'];assert sha(path)==receipt['sha256']
                    source_pins[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
            path=source/'arrays'/w/role/'zarr.json';source_pins[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
        target=np.asarray(zarr.open_array(str(source/'arrays'/w/'inklabels'),mode='r')[y:y+128,x:x+128])>0
        mask=np.asarray(zarr.open_array(str(source/'arrays'/w/'supervision'),mode='r')[y:y+128,x:x+128])>0
        assert mask.sum()==row['supervised_pixels'] and (target&mask).sum()==row['ink_supervised']
        targets.append(np.where(mask,target,False));masks.append(mask)
    np.savez_compressed(a.work/'TRAINING_FIXTURE.npz',raw=np.stack(raw),target=np.stack(targets),mask=np.stack(masks))
    prior=json.loads((a.repo/'research/in-progress/native-common-comparison-20261012/GO.json').read_text())
    checkpoint=next(r['checkpoint'] for r in prior['models'] if r['name']=='dnative')
    assert sha(Path(checkpoint['path']))==checkpoint['sha256']
    plan=dict(status='FROZEN_MECHANICS_CHECK_ONLY',checkpoint=checkpoint,fixture=chosen,
        fixture_selection='First lexicographic eligible negative and two-class core on w035 and w039; class-aware mechanics fixture only, not a training cohort or validation selection',
        source_pins=source_pins,fixture_sha256=sha(a.work/'TRAINING_FIXTURE.npz'),
        implementation={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('matched_training.py')]},
        original_fixture_indices=[0,1],added_fixture_indices=[2,3],
        original_coefficient=.5,learning_rate=.01,momentum=.9,weight_decay=3e-5,gradient_clip=1.,
        completed_steps_per_trajectory=2,bce_smoothing=.5,dice_weight=.25,
        accumulation_tolerance=dict(relative_gradient_l2=1e-4,max_abs_scaled_by_max_reference=1e-4),
        dtype='float32',device='cpu',threads=4,training_accuracy_evaluation=False,
        actual_added_0814_data_qualified=False,small_fixture_is_not_the_training_population=True,
        augmentation='Independently keyed rot90, XY flips, native Z window offset -2..2. Default published augmentation was separately preflighted; this bounded restart fixture is not a full production loader qualification.')
    write(a.work/'TRAINING_GO.json',plan)
    print(json.dumps(dict(status=plan['status'],real_patches=4,updates_per_trajectory=2)),flush=True)


def run(a):
    go=json.loads((a.work/'TRAINING_GO.json').read_text());binding=sha(a.work/'TRAINING_GO.json')
    if (a.work/'TRAINING_QUALIFICATION.json').exists():raise FileExistsError('Already completed')
    for p in [Path(__file__),Path(__file__).with_name('matched_training.py')]:assert sha(p)==go['implementation'][p.name]
    assert sha(a.work/'TRAINING_FIXTURE.npz')==go['fixture_sha256']
    ck=Path(go['checkpoint']['path']);assert sha(ck)==go['checkpoint']['sha256']
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(12012);random.seed(12012);np.random.seed(12012)
    sys.path.insert(0,str(a.repo/'research/lab/reports/fortysixth-experiment/runtime-source'))
    from vesuvius.ink_detection.inference import infer
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.training.losses import create_loss
    infer.load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    infer.load_flat_inference_state=lambda m,s:m.load_state_dict(s,strict=True)
    conf=infer.configure_model(infer.parse_args(['unused.zarr',str(ck),'unused.tif','--no-compile']))
    model=conf.model;model.train();require_sample_independent_model(model)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    initial_sha=tensor_state_hash(initial)
    fixture=np.load(a.work/'TRAINING_FIXTURE.npz',allow_pickle=False)
    norm_counts={}
    for m in model.modules():
        if 'Norm' in type(m).__name__:norm_counts[type(m).__name__]=norm_counts.get(type(m).__name__,0)+1
    cfg=InkConfig.from_mapping(dict(mode='flat',model_type='vesuvius_unet_3d_stem_2d',in_channels=1,
        patch_size=[17,128,128],patch_overlap=.25,patch_min_labeled_coverage=0.,
        datasets=[dict(segments_path='fixture',volume_scale=0)],
        targets=dict(ink=dict(activation='none',out_channels=1)),loss=dict(bce_label_smoothing=.5,dice_label_smoothing=0.),out_dir='unused'))
    original_loss=create_loss(cfg)
    def batch(step):
        xs=[];ys=[];ms=[];schedule=[]
        for stream,indices in [('original',[0,1]),('added',[2,3])]:
            for index,fi in enumerate(indices):
                draw=slot(12012,step,stream,index,[str(fi)])
                rng=np.random.default_rng(draw['augmentation_seed']);offset=int(rng.integers(-2,3));rot=int(rng.integers(4));flip=bool(rng.integers(2))
                x=fixture['raw'][fi,2+offset:19+offset].copy();y=fixture['target'][fi].copy();m=fixture['mask'][fi].copy()
                x=np.rot90(x,rot,axes=(-2,-1));y=np.rot90(y,rot);m=np.rot90(m,rot)
                if flip:x=x[...,::-1];y=y[:,::-1];m=m[:,::-1]
                xs.append(infer.normalize_flat_patch(x,conf.preprocessing));ys.append(y.copy());ms.append(m.copy())
                schedule.append(dict(draw,fixture_index=fi,z_offset=offset,rotation=rot,flip_x=flip))
        return torch.tensor(np.stack(xs)[:,None]),torch.tensor(np.stack(ys)[:,None],dtype=torch.float32),torch.tensor(np.stack(ms)[:,None]),schedule
    def objectives(z,y,m):
        return sum(.5*partial_stream_loss(z[i:i+2],y[i:i+2],m[i:i+2],stream_totals(y[i:i+2],m[i:i+2])) for i in [0,2])
    # Separate formula vs retained objective and microbatch decomposition on
    # finite logits. Unequal masks make naive microbatch means a real negative control.
    x,y,m,_=batch(0);logits=torch.linspace(-1,1,y.numel()).reshape(y.shape).requires_grad_()
    expected=.5*original_loss(logits[:2],torch.cat([y[:2],(~m[:2]).float()],1))+.5*original_loss(logits[2:],torch.cat([y[2:],(~m[2:]).float()],1))
    actual=objectives(logits,y,m);assert torch.allclose(expected,actual,atol=2e-7,rtol=2e-7)
    ga=torch.autograd.grad(actual,logits,retain_graph=True)[0];ge=torch.autograd.grad(expected,logits)[0]
    assert torch.allclose(ga,ge,atol=1e-9,rtol=1e-5)
    naive=sum(.25*original_loss(logits[i:i+1],torch.cat([y[i:i+1],(~m[i:i+1]).float()],1)) for i in range(4))
    assert abs(float(naive.detach()-actual.detach()))>1e-6
    rejections=[]
    try:stream_totals(y,m&False)
    except ValueError:rejections.append('empty_supervision')
    try:require_sample_independent_model(torch.nn.Sequential(torch.nn.BatchNorm2d(1)))
    except ValueError:rejections.append('batch_normalization')
    assert len(rejections)==2
    draws=0
    for step in range(100):
        for index in range(32):
            original_a=slot(42,step,'original',index,['a','b','c'])
            slot(42,step,'added',index,['x','y']);slot(42,step,'added',index,['x','y','z','w'])
            assert original_a==slot(42,step,'original',index,['c','b','a']);draws+=1
    start=time.monotonic();forwards=0
    def grads(step,micro):
        nonlocal forwards
        model.zero_grad(set_to_none=True);x,y,m,schedule=batch(step);loss_total=0.
        if micro:
            for start in [0,2]:
                totals=stream_totals(y[start:start+2],m[start:start+2])
                for i in range(start,start+2):
                    z=model(x[i:i+1]);forwards+=1
                    loss=.5*partial_stream_loss(z,y[i:i+1],m[i:i+1],totals);loss.backward();loss_total+=float(loss.detach())
        else:
            z=model(x);forwards+=1;loss=objectives(z,y,m);loss.backward();loss_total=float(loss.detach())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        return loss_total,schedule
    full_loss,_=grads(0,False);full={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    micro_loss,_=grads(0,True)
    sqdiff=sqref=0.;maxdiff=maxref=0.
    for n,p in model.named_parameters():
        if n not in full:continue
        diff=p.grad-full[n];sqdiff+=float(diff.double().square().sum());sqref+=float(full[n].double().square().sum())
        maxdiff=max(maxdiff,float(diff.abs().max()));maxref=max(maxref,float(full[n].abs().max()))
    rel=(sqdiff/sqref)**.5;scaled=maxdiff/maxref
    assert rel<=1e-4 and scaled<=1e-4,(rel,scaled)
    del full
    write(a.work/'TRAINING_STATUS.json',dict(phase='accumulation_passed',relative_gradient_l2=rel,forwards=forwards))
    def optimizer():return torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=3e-5)
    opt=optimizer();trajectory=[]
    model.load_state_dict(initial)
    for step in [0,1]:
        value,schedule=grads(step,True);norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step()
        trajectory.append(dict(step=step,loss=value,unclipped_gradient_norm=norm,schedule=schedule,model_sha256=tensor_state_hash(model.state_dict())))
        if step==0:
            payload=dict(binding=binding,next_step=1,model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state())
            path=a.work/'mechanics-step1.pth';torch.save(payload,path)
            checkpoint_sha=sha(path)
            write(a.work/'mechanics-step1.json',dict(binding=binding,path=path.name,sha256=checkpoint_sha,bytes=path.stat().st_size))
    completed={k:v.detach().clone() for k,v in model.state_dict().items()}
    complete_momentum=tensor_state_hash({str(i):s['momentum_buffer'] for i,s in opt.state_dict()['state'].items()})
    assert tensor_state_hash(completed)!=initial_sha
    # Fresh objects exercise restoration. This is a save/reload boundary, not
    # a simulated power failure or lost-success-response claim.
    model.load_state_dict(initial);opt=optimizer()
    receipt=json.loads((a.work/'mechanics-step1.json').read_text());assert sha(a.work/receipt['path'])==receipt['sha256']
    restored=torch.load(a.work/receipt['path'],map_location='cpu',weights_only=True);assert restored['binding']==binding
    assert restored['next_step']==1
    model.load_state_dict(restored['model'],strict=True);opt.load_state_dict(restored['optimizer']);torch.set_rng_state(restored['torch_rng'])
    value,schedule=grads(restored['next_step'],True);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    assert value==trajectory[-1]['loss'] and schedule==trajectory[-1]['schedule']
    assert all(torch.equal(v,completed[k]) for k,v in model.state_dict().items())
    assert tensor_state_hash({str(i):s['momentum_buffer'] for i,s in opt.state_dict()['state'].items()})==complete_momentum
    result=dict(status='PASS',go_sha256=binding,parameters=sum(p.numel() for p in model.parameters()),
        model_norm_modules=norm_counts,batch_stochastic_coupling_rejected=True,real_native_patches=4,
        partial_loss_matches_retained_objective=True,naive_microbatch_average_rejected=True,
        full_vs_microbatch=dict(full_loss=full_loss,micro_loss=micro_loss,relative_gradient_l2=rel,
                               maximum_absolute_gradient_difference=maxdiff,scaled_maximum_difference=scaled),
        original_schedule_checks=draws,negative_controls=rejections,completed_updates=2,replayed_updates=1,
        model_and_optimizer_resume_bitwise_equal=True,trained_model_forwards=forwards,
        checkpoint_sha256=checkpoint_sha,trajectory=trajectory,seconds=time.monotonic()-start,
        original_released_checkpoint_unchanged=sha(ck)==go['checkpoint']['sha256'],
        checkpoint_scope='Mechanics only, never an evaluated model or learning result',
        cuda_tested=False,paid_compute=False,held_out_predictions=0,
        limits=['Not CUDA memory/speed or AMP qualification','Actual 0814 population and mapping remain pending',
                'Final sampler, source quotas and objective coefficients require full data freeze',
                'Counter-based basic augmentation tested here; published full augmentation covered by prior synthetic preflight'])
    write(a.work/'TRAINING_QUALIFICATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='trajectory'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    (prepare if a.mode=='prepare' else run)(a)
