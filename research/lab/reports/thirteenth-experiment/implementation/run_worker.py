"""Root-only32-update conservative control with isolated fixed-risk measurements."""
import argparse
import gc
import hashlib
import json
import math
import platform
import random
import resource
import signal
import socket
import time
from pathlib import Path
import numpy as np
from control_common import ARMS,UPDATES,load_plan,module,verify,record,write_json,require,training_region,state_hash,average,eligible
from training_mechanics import augment,losses as hard_losses,arm_loss
from smoothing import smoothed_losses


def offline(*args,**kwargs): raise RuntimeError('Network forbidden in conservative control')
socket.create_connection=offline; socket.socket.connect=offline; socket.socket.connect_ex=offline


def rss(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def tree_hash(value,torch):
    digest=hashlib.sha256()
    def frame(tag,data=b''):
        digest.update(tag.encode()+b'\0'+len(data).to_bytes(8,'big')+data)
    def visit(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous(); frame('tensor_metadata',json.dumps([list(a.shape),str(a.dtype)]).encode()); frame('tensor_data',a.numpy().tobytes())
        elif isinstance(v,np.ndarray):
            frame('numpy_metadata',json.dumps([list(v.shape),str(v.dtype)]).encode()); frame('numpy_data',v.tobytes(order='C'))
        elif isinstance(v,dict):
            frame('dict',str(len(v)).encode())
            for key in sorted(v,key=lambda k:(type(k).__name__,str(k))): visit(key); visit(v[key])
        elif isinstance(v,(list,tuple)):
            frame(type(v).__name__,str(len(v)).encode())
            for item in v: visit(item)
        else: frame(type(v).__name__,json.dumps(v,sort_keys=True,allow_nan=False).encode())
    visit(value); return digest.hexdigest()


def isolation_state(model,optimizer,torch):
    return dict(model=state_hash(model.state_dict()),optimizer=tree_hash(optimizer.state_dict(),torch),
        gradients=tree_hash({name:p.grad for name,p in model.named_parameters()},torch),
        modes=[layer.training for layer in model.modules()])


def mode_guard(model,meta,torch):
    require(meta['embedded_config']['targets']['ink']['activation']=='none' and not list(model.named_buffers()),'Mode/buffer contract differs')
    for layer in model.modules():
        require(not isinstance(layer,torch.nn.modules.batchnorm._BatchNorm) and not getattr(layer,'track_running_stats',False),'Running normalization unsupported')
        if isinstance(layer,torch.nn.modules.dropout._DropoutNd): require(layer.p==0,'Active dropout unsupported')
        if type(layer).__name__ in ['DropPath','StochasticDepth']:
            require(getattr(layer,'drop_prob',getattr(layer,'p',0))==0,'Active stochastic depth unsupported')
    return dict(no_running_buffers=True,no_active_stochastic_layers=True,measurement_mode='unchanged training mode')


def batch(step,regions,runtime,torch):
    images=[]; masks=[[],[]]; targets=[[],[]]; exposures=[]
    for patch in step['patches']:
        r=patch['region']; data,m,t,p,pt=regions[r]; x,y,w,h=patch['bbox']; code=patch['augmentation']
        image=np.ascontiguousarray(runtime['normalize_robust'](data['surface'][2:19,y:y+h,x:x+w]),dtype=np.float32)
        require(np.isfinite(image).all(),'Normalization not finite'); images.append(augment(image,code)[None]); exposure=dict(region=r)
        for i,(mask,target,name) in enumerate([(m,t,'manual'),(p,pt,'pseudo')]):
            mm=augment(mask[y:y+h,x:x+w],code); tt=augment(target[y:y+h,x:x+w],code)
            masks[i].append(mm[None]); targets[i].append(tt[None])
            exposure[name]=dict(positive=int((mm&(tt==1)).sum()),negative=int((mm&(tt==0)).sum()))
        require(min(exposure['manual'].values())>=256,'Manual support gate changed'); exposures.append(exposure)
    tensors=[torch.from_numpy(np.stack(v)) for v in [images,*masks,*targets]]
    image,m,p,t,pt=tensors
    identity=state_hash(dict(input=image,manual_mask=m,manual_hard_target=t,pseudo_mask=p,pseudo_hard_target=pt))
    return (image,m,t,p,pt),identity,exposures


def forward(model,image,torch):
    before=state_hash(dict(input=image)); value=model(image)['ink']; z=value[0] if isinstance(value,(list,tuple)) else value
    require(z.dtype==torch.float32 and tuple(z.shape)==(2,1,128,128) and bool(torch.isfinite(z).all()),'Logit contract differs')
    require(state_hash(dict(input=image))==before,'Model mutated input'); return z


def terms(z,teacher,batch_tensors,arm,preservation,torch):
    _,m,t,p,pt=batch_tensors; u,pseudo,balanced=smoothed_losses(z,m,t,p,pt,.5)
    dm,dp=preservation.preservation_losses(z,teacher,m,p); base=arm_loss(arm.removesuffix('_preserve'),u,pseudo,balanced)
    total=base+(dm if arm=='A_preserve' else dm+.25*dp)
    with torch.no_grad():
        hu,hp,_=hard_losses(z,m,t,p,pt); diff=(torch.sigmoid(z.detach().double())-torch.sigmoid(teacher.double())).abs()
        values=dict(loss_U=float(u.detach()),loss_P=float(pseudo.detach()),loss_R=float(balanced.detach()),
            loss_D_M=float(dm.detach()),loss_D_P=float(dp.detach()),base=float(base.detach()),full=float(total.detach()),
            hard_base=float(arm_loss(arm.removesuffix('_preserve'),hu,hp,balanced).detach()),
            probability_drift_manual=float(diff[m].mean()),probability_drift_pseudo=float(diff[p].mean()))
    require(all(math.isfinite(v) for v in values.values()),'Nonfinite loss/drift scalar')
    return total,values


def measure(model,optimizer,batches,teachers,indices,arm,preservation,torch):
    before=isolation_state(model,optimizer,torch); rng=torch.random.get_rng_state().clone()
    numpy_rng=np.random.get_state(); python_rng=random.getstate(); rows=[]; logits=[]
    with torch.no_grad():
        for i in indices:
            z=forward(model,batches[i][0],torch); _,values=terms(z,teachers[i],batches[i],arm,preservation,torch)
            rows.append(values); logits.append(z.numpy().copy())
    after=isolation_state(model,optimizer,torch)
    rng_same=torch.equal(rng,torch.random.get_rng_state()) and tree_hash(numpy_rng,torch)==tree_hash(np.random.get_state(),torch) and python_rng==random.getstate()
    torch.random.set_rng_state(rng); np.random.set_state(numpy_rng); random.setstate(python_rng)
    require(before==after and rng_same,'Measurement altered model/optimizer/gradients/modes/RNG')
    return np.stack(logits),rows,dict(before=before,after=after,rng_unchanged=True,no_grad=True,model_mode_unchanged=True)


def execute(args):
    started=time.perf_counter(); signal.alarm(180)
    resource.setrlimit(resource.RLIMIT_CPU,(600,600)); resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(200*1024**2,200*1024**2))
    plan,path=load_plan(args.plan,execute=True); planpin=record(path); args.output_dir.mkdir(parents=True,exist_ok=False)
    import torch
    require(platform.python_version()==plan['runtime']['python'] and np.__version__==plan['runtime']['numpy'] and torch.__version__==plan['runtime']['torch'],'Pinned runtime differs')
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261008)
    helper=module(plan['helper'],'conservative_model_contract'); preservation=module(plan['shared_sources']['preservation.py'],'conservative_preservation')
    runtime=helper.load_runtime(); model,modelmeta=helper.build_model(Path(plan['initializer']['path']),plan['initializer'],runtime)
    mode=mode_guard(model,modelmeta,torch); model.train(); initial=state_hash(model.state_dict())
    require(initial==plan['historical_initial_state_sha256'],'Initializer changed')
    schedule=json.loads(verify(plan['schedule']).read_text()); histories={}
    for arm in ARMS: histories[arm]=[json.loads(s) for s in verify(plan['historical'][arm]['steps']).read_text().splitlines()]
    cache=np.load(verify(plan['historical'][args.arm]['teacher']),mmap_mode='r',allow_pickle=False)
    other=np.load(verify(plan['historical'][next(a for a in ARMS if a!=args.arm)]['teacher']),mmap_mode='r',allow_pickle=False)
    require(cache.dtype==np.float32 and cache.shape==(32,2,1,128,128) and np.isfinite(cache).all() and np.array_equal(cache.view(np.uint32),other.view(np.uint32)),'Frozen teacher cache differs')
    regions=[training_region(c) for c in plan['adaptation']]; batches=[]; identities=[]; exposures=[]; teachers=[]
    for i,step in enumerate(schedule):
        tensors,identity,exposure=batch(step,regions,runtime,torch)
        for arm in ARMS:
            require(histories[arm][i]['step']==i+1 and histories[arm][i]['patches']==step['patches'] and histories[arm][i]['exposures']==exposure and histories[arm][i]['input_target_sha256']==identity,'Exact historical batch changed')
        batches.append(tensors); identities.append(identity); exposures.append(exposure)
        teachers.append(torch.from_numpy(np.array(cache[i],copy=True))); require(not teachers[-1].requires_grad,'Cached teacher has gradients')
    with torch.no_grad(): baseline_rows=[terms(t,t,b,args.arm,preservation,torch)[1] for t,b in zip(teachers,batches)]
    baseline=dict(scope='cache-derived initializer risk; no new model forward',per_batch=baseline_rows,equal_batch_means=average(baseline_rows))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0)
    pre=np.lib.format.open_memmap(args.output_dir/'student-preupdate-logits.npy',mode='w+',dtype=np.float32,shape=(32,2,1,128,128))
    post=np.lib.format.open_memmap(args.output_dir/'student-postupdate-logits.npy',mode='w+',dtype=np.float32,shape=(32,2,1,128,128))
    fixed=np.lib.format.open_memmap(args.output_dir/'fixed-risk-logits.npy',mode='w+',dtype=np.float32,shape=(4,32,2,1,128,128))
    evaluations={}; logpath=args.output_dir/'steps.jsonl'
    with logpath.open('x') as stream:
        for i,step in enumerate(schedule):
            clock=time.perf_counter(); tensors=batches[i]; image,m,t,p,pt=tensors
            require(state_hash(dict(input=image,manual_mask=m,manual_hard_target=t,pseudo_mask=p,pseudo_hard_target=pt))==identities[i],'Training batch mutated')
            optimizer.zero_grad(set_to_none=True); z=forward(model,image,torch); pre[i]=z.detach().numpy(); z.retain_grad()
            total,values=terms(z,teachers[i],tensors,args.arm,preservation,torch)
            if i==0:
                require(np.array_equal(z.detach().numpy().view(np.uint32),cache[i].view(np.uint32)),'First logits differ from certified initializer cache')
                require(abs(values['base']-histories[args.arm][0]['original_objective'])<=1e-7 and abs(values['loss_U']-histories[args.arm][0]['loss_U'])<=1e-7 and abs(values['loss_P']-histories[args.arm][0]['loss_P'])<=1e-7,'First smoothed losses differ')
            total.backward(); active=m if args.arm=='A_preserve' else (m|p)
            require(bool((z.grad[~active]==0).all()),'Unknown output locations contributed')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            require(float(norm)>0 and bool(torch.isfinite(norm)),'Invalid gradient')
            if i==0: require(abs(float(norm)-histories[args.arm][0]['preclip_global_gradient_norm'])<=1e-6,'Historical first gradient differs')
            before={name:value.detach().clone() for name,value in model.named_parameters()}; optimizer.step()
            update_l2=sum(float((value.detach().double()-before[name].double()).square().sum()) for name,value in model.named_parameters())**.5
            require(update_l2>0 and all(bool(torch.isfinite(v).all()) for v in model.parameters()),'No finite parameter update'); del before
            measured,post_rows,guard=measure(model,optimizer,batches,teachers,[i],args.arm,preservation,torch); post[i]=measured[0]
            if i==0:
                proof=plan['first_step_probe'][args.arm]; expected=np.load(verify(proof['post_logits']),allow_pickle=False)
                require(state_hash(model.state_dict())==proof['post_state_sha256'] and expected.dtype==np.float32 and np.array_equal(measured[0].view(np.uint32),expected.view(np.uint32)),'Twelfth1e-6 first post state/logits differ')
                write_json(args.output_dir/'first-step-sentinel.json',dict(initial_state_sha256=initial,post_state_sha256=proof['post_state_sha256'],
                    input_target_sha256=identities[0],historical_first_logits_bitwise_equal=True,twelfth_post_logits_bitwise_equal=True,
                    gradient_norm=float(norm),pre=values,post=post_rows[0]))
            if step['step'] in UPDATES:
                fz,fr,fg=measure(model,optimizer,batches,teachers,list(range(32)),args.arm,preservation,torch)
                require(np.array_equal(fz[i].view(np.uint32),measured[0].view(np.uint32)),'Fixed risk and immediate post logits differ')
                fixed[UPDATES.index(step['step'])]=fz
                evaluations[str(step['step'])]=dict(per_batch=fr,equal_batch_means=average(fr),isolation=fg,
                    current_batch_post_bitwise_equal=True,state_sha256=fg['after']['model'])
            row=dict(step=i+1,patches=step['patches'],exposures=exposures[i],input_target_sha256=identities[i],pre=values,post=post_rows[0],
                preclip_global_gradient_norm=float(norm),parameter_update_l2=update_l2,immediate_measurement_isolation=guard,
                unknown_output_gradient_max=0.,wall_seconds=time.perf_counter()-clock,peak_rss_bytes=rss())
            stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush(); require(rss()<=plan['limits']['rss_acceptance_bytes'],'RSS acceptance failed')
            del z,total
    for array in [pre,post,fixed]: array.flush()
    del pre,post,fixed; final=state_hash(model.state_dict()); require(final!=initial,'Final state unchanged')
    checkpoint=args.output_dir/'final-state.pth'; torch.save(model.state_dict(),checkpoint); del optimizer; model.zero_grad(set_to_none=True); gc.collect()
    persisted=torch.load(checkpoint,map_location='cpu',weights_only=True); expected=model.state_dict()
    require(isinstance(persisted,dict) and set(persisted)==set(expected),'Persisted keys differ')
    for key,value in persisted.items():
        require(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.shape==expected[key].shape and value.dtype==expected[key].dtype and bool(torch.isfinite(value).all()) and torch.equal(value,expected[key]),'Persisted tensor differs: '+key)
    storage=helper.verify_storage_contract(persisted,expected,torch); require(state_hash(persisted)==final,'Persisted semantic state differs')
    strict=model.load_state_dict(persisted,strict=True); require(not strict.missing_keys and not strict.unexpected_keys,'Persisted strict load failed')
    del persisted,expected; gc.collect(); load_plan(path,execute=True); require(record(path)==planpin,'Plan changed')
    require(sum(f.stat().st_size for f in args.output_dir.rglob('*') if f.is_file())<=plan['limits']['output_bytes'],'Output cap failed')
    result=dict(status='PASS',arm=args.arm,plan=planpin,initial_state_sha256=initial,final_state_sha256=final,checkpoint=record(checkpoint),
        model_contract=modelmeta,mode_guard=mode,steps_log=record(logpath),teacher_cache=plan['historical'][args.arm]['teacher'],
        raw_logits=dict(preupdate=record(args.output_dir/'student-preupdate-logits.npy'),postupdate=record(args.output_dir/'student-postupdate-logits.npy'),fixed_risk=record(args.output_dir/'fixed-risk-logits.npy')),
        first_step_sentinel=record(args.output_dir/'first-step-sentinel.json'),batch_fingerprints=identities,
        baseline_fixed_evaluation=baseline,fixed_evaluation=evaluations,
        eligible=eligible(baseline['equal_batch_means'],evaluations['32']['equal_batch_means']),
        persisted_validation=dict(weights_only=True,exact_tensor_values=True,strict=True,storage_contract=storage),
        actual_calls=dict(student_forwards=192,teacher_forwards=0,backwards=32,optimizer_steps=32),
        scope='Fixed adaptation risk only; no calibration, retention or heldout inference',
        wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss(),threads=4,torch_version=torch.__version__)
    write_json(args.output_dir/'result.json',result); signal.alarm(0)
    print(json.dumps(dict(status='PASS',arm=args.arm,eligible=result['eligible'],wall_seconds=result['wall_seconds'],peak_rss_bytes=rss())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--arm',required=True,choices=ARMS); parser.add_argument('--output-dir',required=True,type=Path)
    execute(parser.parse_args())
