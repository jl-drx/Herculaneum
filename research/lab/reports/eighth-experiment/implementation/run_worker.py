"""One fixed CPU arm: updates, development lock, then fresh predictions."""
import argparse
import gc
import json
import resource
import signal
import socket
import time
from pathlib import Path
import numpy as np
from experiment_common import load_plan, module, verify, record, write_json, require, training_region, evaluation, state_hash
from training_mechanics import augment, losses, arm_loss


def offline(*args,**kwargs): raise RuntimeError('Network disabled in frozen CPU worker')
socket.create_connection=offline
socket.socket.connect=offline
socket.socket.connect_ex=offline


def rss(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def infer(model, case, runtime, directory):
    data,_,_=evaluation(case); directory.mkdir(parents=True,exist_ok=False)
    scores,meta,evaluated=runtime['reference'].infer_surface(model,data['surface'],2,'forward',runtime['normalize_robust'],None)
    np.save(directory/'scores.npy',scores); np.save(directory/'evaluated-mask.npy',evaluated)
    result=dict(scores=record(directory/'scores.npy'),evaluated=record(directory/'evaluated-mask.npy'),inference=meta)
    write_json(directory/'metadata.json',result); return result


def train(model, plan, runtime, helper, arm, directory):
    import torch
    started=time.perf_counter(); regions=[training_region(c) for c in plan['adaptation']]
    schedule=json.loads(verify(plan['schedule']).read_text()); require(len(schedule)==32,'Schedule length changed')
    initial_state=state_hash(model.state_dict()); initial_parameters=state_hash(dict(model.named_parameters()))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    model.train(); logfile=directory/'steps.jsonl'; model.zero_grad(set_to_none=True)
    with logfile.open('x') as stream:
        for step in schedule:
            clock=time.perf_counter(); images=[]; masks=[[],[]]; targets=[[],[]]; exposures=[]
            for patch in step['patches']:
                r=patch['region']; data,m,t,p,pt=regions[r]; x,y,w,h=patch['bbox']; code=patch['augmentation']
                raw=data['surface'][2:19,y:y+h,x:x+w]
                image=np.ascontiguousarray(runtime['normalize_robust'](raw),dtype=np.float32)
                require(np.isfinite(image).all(),'Normalization not finite'); images.append(augment(image,code)[None])
                exposure=dict(region=r)
                for i,(mask,target,name) in enumerate([(m,t,'manual'),(p,pt,'pseudo')]):
                    mm=augment(mask[y:y+h,x:x+w],code); tt=augment(target[y:y+h,x:x+w],code)
                    masks[i].append(mm[None]); targets[i].append(tt[None])
                    exposure[name]=dict(positive=int((mm&(tt==1)).sum()),negative=int((mm&(tt==0)).sum()))
                require(min(exposure['manual'].values())>=256,'Manual anchor gate changed'); exposures.append(exposure)
            tensor=torch.from_numpy(np.stack(images)); m,p=[torch.from_numpy(np.stack(v)) for v in masks]
            t,pt=[torch.from_numpy(np.stack(v)) for v in targets]
            optimizer.zero_grad(set_to_none=True); output=model(tensor)['ink']
            logits=output[0] if isinstance(output,(list,tuple)) else output
            require(tuple(logits.shape)==(2,1,128,128) and bool(torch.isfinite(logits).all()),'Output contract differs')
            logits.retain_grad(); u,pseudo,balanced=losses(logits,m,t,p,pt); total=arm_loss(arm,u,pseudo,balanced)
            require(bool(torch.isfinite(total)),'Loss not finite'); total.backward()
            active=m if arm=='A' else (m|p)
            require(bool((logits.grad[~active]==0).all()),'Unknown output logits contributed to loss')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            require(bool(torch.isfinite(norm)) and float(norm)>0,'Zero/nonfinite gradient')
            probe_name,probe=next((name,p) for name,p in model.named_parameters() if p.grad is not None and bool((p.grad!=0).any()))
            index=int(probe.grad.detach().abs().reshape(-1).argmax()); probe_before=float(probe.detach().reshape(-1)[index])
            optimizer.step(); probe_after=float(probe.detach().reshape(-1)[index])
            require(probe_before!=probe_after,'No measurable parameter update')
            require(rss()<=plan['limits']['rss_acceptance_bytes'],'Peak RSS exceeds acceptance')
            row=dict(step=step['step'],patches=step['patches'],exposures=exposures,
                loss_U=float(u.detach()),loss_P=float(pseudo.detach()),loss_R=float(balanced.detach()),
                total_loss=float(total.detach()),preclip_global_gradient_norm=float(norm),
                unknown_output_gradient_max=0.,parameter_update_probe=dict(name=probe_name,index=index,before=probe_before,after=probe_after),
                wall_seconds=time.perf_counter()-clock,peak_rss_bytes=rss())
            stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Final weights nonfinite')
    final_state=state_hash(model.state_dict()); final_parameters=state_hash(dict(model.named_parameters()))
    require(initial_parameters!=final_parameters,'No actual parameter change')
    checkpoint=directory/'final-state.pth'; torch.save(model.state_dict(),checkpoint)
    del optimizer; model.zero_grad(set_to_none=True); gc.collect()
    persisted=torch.load(checkpoint,map_location='cpu',weights_only=True); expected=model.state_dict()
    require(isinstance(persisted,dict) and set(persisted)==set(expected),'Persisted state keys differ')
    for key,value in persisted.items():
        require(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.shape==expected[key].shape
            and value.dtype==expected[key].dtype and bool(torch.isfinite(value).all()) and torch.equal(value,expected[key]),'Persisted tensor differs: '+key)
    storage=helper.verify_storage_contract(persisted,expected,torch)
    require(state_hash(persisted)==final_state,'Persisted state semantic hash differs')
    strict=model.load_state_dict(persisted,strict=True)
    require(not strict.missing_keys and not strict.unexpected_keys,'Persisted strict load failed')
    del persisted,expected; gc.collect()
    summary=dict(initial_state_sha256=initial_state,final_state_sha256=final_state,
        initial_parameter_sha256=initial_parameters,final_parameter_sha256=final_parameters,
        checkpoint=record(checkpoint),steps=32,steps_log=record(logfile),schedule=plan['schedule'],
        exposure=plan['exposure'],training_seconds=time.perf_counter()-started,
        checkpoint_format='raw state dictionary; weights_only=True compatible; no optimizer or intermediate weights',
        unsmoothed_targets=True,inherited_checkpoint_bce_label_smoothing=.5,
        persisted_validation=dict(weights_only=True,exact_tensor_values=True,missing_keys=[],unexpected_keys=[],
            semantic_state_sha256=final_state,storage_contract=storage))
    write_json(directory/'training.json',summary)
    del regions,images,masks,targets,tensor,m,p,t,pt,output,logits,u,pseudo,balanced,total
    model.zero_grad(set_to_none=True); gc.collect(); model.eval(); return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--arm',choices=['frozen_seed42','A','B','C'],required=True); parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); started=time.perf_counter()
    signal.alarm(120); resource.setrlimit(resource.RLIMIT_CPU,(360,360)); resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(200*1024**2,200*1024**2))
    plan,planpath=load_plan(args.plan,execute=True); plan_record=record(planpath); args.output_dir.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261008)
    helper=module(plan['helper'],'eighth_model_contract'); cutoff=module(plan['cutoff'],'eighth_cutoff'); runtime=helper.load_runtime()
    model,modelmeta=helper.build_model(Path(plan['initializer']['path']),plan['initializer'],runtime)
    model.eval()
    initialization=state_hash(model.state_dict()); training=None
    if args.arm!='frozen_seed42': training=train(model,plan,runtime,helper,args.arm,args.output_dir)
    maps={}
    for case in plan['development']:
        maps[case['id']]=case['baseline_maps'] if args.arm=='frozen_seed42' else infer(model,case,runtime,args.output_dir/'development'/case['id'])
    thresholds={}
    for case in plan['development']:
        _,labels,primary=evaluation(case); scores=np.load(verify(maps[case['id']]['scores']),allow_pickle=False)
        evaluated=np.load(verify(maps[case['id']]['evaluated']),allow_pickle=False)
        require(scores.dtype==np.float32 and scores.shape==primary.shape and evaluated.dtype==np.bool_, 'Development map differs')
        require(np.all(evaluated[primary]) and np.isfinite(scores[primary]).all(),'Development primary not fully evaluated')
        thresholds[case['id']]=dict(selection=cutoff.choose_development_threshold(scores[primary&~labels]),maps=maps[case['id']])
    lock=dict(arm=args.arm,plan=plan_record,development=thresholds,
        primary=max(v['selection']['threshold'] for v in thresholds.values()),
        secondary=thresholds['pherc1667-w029']['selection']['threshold'],fresh_inference_started=False,
        scope='Only fixed development negatives; no PHerc0139 labels or fresh values used')
    lockpath=args.output_dir/'thresholds.json'; write_json(lockpath,lock); lock_record=record(lockpath)
    fresh={case['id']:infer(model,case,runtime,args.output_dir/'fresh'/case['id']) for case in plan['fresh']}
    require(record(lockpath)==lock_record,'Threshold lock changed'); load_plan(planpath,execute=True)
    require(record(planpath)==plan_record,'Plan changed'); require(rss()<=plan['limits']['rss_acceptance_bytes'],'Peak RSS acceptance failed')
    size=sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file()); require(size<=plan['limits']['output_bytes'],'Output cap failed')
    write_json(args.output_dir/'result.json',dict(status='PASS',arm=args.arm,plan=plan_record,initializer=plan['initializer'],
        initial_state_sha256=initialization,model_contract=modelmeta,training=training,threshold_lock=lock_record,development=maps,fresh=fresh,
        wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss(),torch_version=torch.__version__,threads=4,
        cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime))
    signal.alarm(0); print(json.dumps(dict(status='PASS',arm=args.arm,wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss())))


if __name__=='__main__': main()
