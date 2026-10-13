"""Standalone timing/restart qualification. No network, real CT or held-out data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from native_model import NativeNet, masked_core_loss

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def fixture():
    g=np.random.default_rng(23011)
    x=g.integers(0,256,size=(8,28,256,256),dtype=np.uint8)
    y=g.integers(0,2,size=(8,128,128),dtype=np.uint8)
    mask=g.random((8,128,128))>.15
    return x,y,mask

def batch(data,rng):
    x,y,mask=data
    counts=mask.sum((1,2));idx=rng.choice(len(x),size=4,replace=True,p=counts/counts.sum())
    xx=(torch.tensor(x[idx],dtype=torch.float32)/255-.5)/.25
    yy=torch.tensor(y[idx],dtype=torch.float32);mm=torch.tensor(mask[idx],dtype=torch.float32)
    for b in range(4):
        k,flip=int(rng.integers(4)),int(rng.integers(2))
        for a in (xx,yy,mm):
            v=torch.rot90(a[b],k,dims=(-2,-1))
            if flip:v=torch.flip(v,dims=(-1,))
            a[b]=v.clone()
    return xx,yy,mm,idx.tolist()

def step(model,opt,data,rng,device):
    model.train();t=time.perf_counter();x,y,m,idx=batch(data,rng);prepared=time.perf_counter()
    x,y,m=(a.to(device) for a in (x,y,m));sync(device);transferred=time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss=masked_core_loss(model(x),y,m)
    if not torch.isfinite(loss):raise ValueError('Non-finite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
    sync(device);end=time.perf_counter()
    return dict(loss=float(loss.detach().cpu()),picked=idx,host_prepare_s=prepared-t,
                transfer_s=transferred-prepared,device_compute_s=end-transferred,wall_s=end-t)

def checkpoint(path,model,opt,rng,device):
    sync(device);t=time.perf_counter()
    obj=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
             optimizer=opt.state_dict(),batch_rng=copy.deepcopy(rng.bit_generator.state),
             torch_cpu_rng=torch.get_rng_state(),
             torch_cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
    tmp=path.with_suffix('.partial')
    with tmp.open('xb') as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return dict(bytes=path.stat().st_size,sha256=sha(path),seconds=time.perf_counter()-t)

def restart(path,receipt,device):
    assert path.stat().st_size==receipt['bytes'] and sha(path)==receipt['sha256']
    obj=torch.load(path,map_location='cpu',weights_only=True)
    model=NativeNet().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    model.load_state_dict(obj['model']);opt.load_state_dict(obj['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=obj['batch_rng']
    torch.set_rng_state(obj['torch_cpu_rng'])
    if device.type=='cuda':torch.cuda.set_rng_state_all(obj['torch_cuda_rng'])
    return model,opt,rng

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def run(args,out):
    torch.set_num_threads(1 if args.mode=='cpu-self-test' else 4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cpu' if args.mode=='cpu-self-test' else 'cuda:0')
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU fallback')
        if torch.__version__.split('+')[0]!='2.12.0' or torch.version.cuda!='12.6':
            raise RuntimeError('Pinned torch2.12.0/CUDA12.6 runtime required')
        if torch.cuda.device_count()!=1:raise RuntimeError('One visible GPU required')
    torch.manual_seed(17);np.random.seed(17)
    data=fixture();rng=np.random.default_rng(17)
    model=NativeNet().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    assert sum(p.numel() for p in model.parameters())==122513
    assert model(torch.zeros(4,28,256,256,device=device)).shape==(4,128,128)
    warmup,timed=(1,2) if args.mode=='cpu-self-test' else (8,32)
    for _ in range(warmup):step(model,opt,data,rng,device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    rows=[step(model,opt,data,rng,device) for _ in range(timed)]
    memory=None if device.type=='cpu' else dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
         peak_reserved_bytes=torch.cuda.max_memory_reserved(device),total_bytes=torch.cuda.get_device_properties(device).total_memory)
    path=out/'restart.pt';receipt=checkpoint(path,model,opt,rng,device)
    control=[step(model,opt,data,rng,device) for _ in range(2)]
    control_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    control_opt=copy.deepcopy(opt.state_dict());control_batch_rng=copy.deepcopy(rng.bit_generator.state)
    control_torch_rng=torch.get_rng_state().clone()
    control_cuda_rng=[v.clone() for v in torch.cuda.get_rng_state_all()] if device.type=='cuda' else []
    del model,opt
    model,opt,rng=restart(path,receipt,device)
    resumed=[step(model,opt,data,rng,device) for _ in range(2)]
    checks=dict(losses_and_batch_draws=all(a['loss']==b['loss'] and a['picked']==b['picked'] for a,b in zip(control,resumed)),
                model=equal(control_model,model.state_dict()),optimizer=equal(control_opt,opt.state_dict()),
                batch_rng=control_batch_rng==rng.bit_generator.state,torch_cpu_rng=torch.equal(control_torch_rng,torch.get_rng_state()),
                torch_cuda_rng=equal(control_cuda_rng,torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
    assert all(checks.values()),'Same-backend restart failed'
    model.eval();t=time.perf_counter();validation_examples=8 if args.mode=='cpu-self-test' else 139
    with torch.no_grad():
        for start in range(0,validation_examples,4):
            n=min(4,validation_examples-start)
            x,y,m,_=batch(data,rng);pred=model(x[:n].to(device))
            assert torch.isfinite(pred).all()
    sync(device);validation_seconds=time.perf_counter()-t
    versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                  cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cpu_threads=torch.get_num_threads(),
                  gpu=None if device.type=='cpu' else torch.cuda.get_device_name(device),
                  capability=None if device.type=='cpu' else list(torch.cuda.get_device_capability(device)))
    wall=sum(r['wall_s'] for r in rows)
    return dict(status='PASS',scope='CPU harness check only' if device.type=='cpu' else 'GPU synthetic timing/restart only',
        versions=versions,warmup_steps=warmup,timed_steps=timed,batch_size=4,shape=[4,28,256,256],scored_shape=[4,128,128],
        precision='float32; TF32 off; deterministic algorithms required',steps_per_second=timed/wall,
        timings={k:dict(mean=float(np.mean([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])))
                 for k in ['host_prepare_s','transfer_s','device_compute_s','wall_s']},steps=rows,memory=memory,
        checkpoint=receipt,restart_checks=checks,validation=dict(examples=validation_examples,seconds=validation_seconds,
            scope='Synthetic forward-loop overhead; not real data assembly or ranking metrics'),
        source=dict(model_sha256=sha(Path(__file__).with_name('native_model.py')),benchmark_sha256=sha(Path(__file__))),
        real_data_used=False,scientific_training_performed=False,private_export_verified=False,
        independent_shutdown_verified=False,launch_readiness=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cpu-self-test','cuda'],required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:result=run(args,args.output)
    except Exception as exc:
        result=dict(status='FAIL',error_type=type(exc).__name__,error=str(exc),launch_readiness=False)
        raise
    finally:
        result.update(mode=args.mode,utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_seconds=time.perf_counter()-started)
        (args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','scope','steps_per_second','restart_checks','launch_readiness']}))

if __name__=='__main__':main()
