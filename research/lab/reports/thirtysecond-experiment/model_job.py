"""Prepare pinned normalized inputs, then one bounded offline seed42 CPU job."""
import argparse,gc,os,resource,signal,socket,time
import numpy as np
from common import *

def offline(*args,**kwargs):raise RuntimeError('Network prohibited in CPU model stage')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
sys.path.insert(0,str(LAB))
import run_ink_baseline as adapter

def state_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0'+json.dumps(list(a.shape)).encode()+b'\0');h.update(a.tobytes())
    return h.hexdigest()

def prepare():
    check_freeze('POOL_GO.json')
    pooled=json.loads((HERE/'pooled-v1/COMPLETE.json').read_text())
    _,_,normalize,source_manifest_sha=adapter.load_focused_sources()
    out=HERE/'normalized-v1';out.mkdir(exist_ok=False);records=[]
    for name,c in pooled['contexts'].items():
        surface=np.load(verify(c['array']),allow_pickle=False)
        require(surface.shape==(21,128,128) and surface.dtype==np.float32,'Pooled shape/dtype')
        for start in range(5):
            raw=surface[start:start+17];require(np.any(raw),'Entire input window is zero')
            x=np.ascontiguousarray(normalize(raw),dtype=np.float32)
            require(x.shape==(17,128,128) and np.isfinite(x).all(),'Normalized input shape/finite')
            path=out/f'{name}-{start}.npy';np.save(path,x,allow_pickle=False)
            records.append(dict(context=name,start=start,array=pin(path),data_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                                min=float(x.min()),max=float(x.max()),pooled_source=c['array']))
    require(len(records)==20,'Input window count')
    save(out/'COMPLETE.json',dict(status='PASS',source_manifest_sha256=source_manifest_sha,inputs=records,
                                 scientific_model_forwards=0,created_at_utc=now()))
    print(json.dumps(dict(status='PASS',normalized_inputs=20,model_forwards=0)))

def worker():
    import torch
    started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('Model wall budget')));signal.alarm(600)
    freeze=check_freeze('MODEL_GO.json')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261009)
    require(not torch.cuda.is_available(),'CPU environment expected')
    config,factory,normalize,manifest_sha=adapter.load_focused_sources()
    checkpoint=verify(freeze['checkpoint']);model,meta=adapter.build_checkpoint_model(checkpoint,'seed42',config,factory)
    require(not model.training and all(p.grad is None for p in model.parameters()),'Eval/gradient contract')
    digest=state_digest(model);inputs=json.loads(verify(freeze['inputs']).read_text())
    out=HERE/'inference-v1';out.mkdir(exist_ok=False)
    save(out/'STARTED.json',dict(created_at_utc=now(),GO=pin(HERE/'MODEL_GO.json'),state_sha256=digest))
    records=[];attempted=0
    try:
        with torch.inference_mode():
            for item in inputs['inputs']:
                require(attempted<20,'Forward budget')
                a=np.load(verify(item['array']),allow_pickle=False)
                require(hashlib.sha256(a.tobytes()).hexdigest()==item['data_sha256'],'Input data differs')
                tensor=torch.from_numpy(a)[None,None];require(tuple(tensor.shape)==(1,1,17,128,128) and tensor.dtype==torch.float32,'Model tensor contract')
                attempted+=1;save(out/'PROGRESS.json',dict(attempted_forwards=attempted,completed_forwards=len(records)))
                t=time.monotonic();output=model(tensor);logits=output['ink']
                if isinstance(logits,(list,tuple)):logits=logits[0]
                require(tuple(logits.shape)==(1,1,128,128) and logits.dtype==torch.float32 and bool(torch.isfinite(logits).all()),'Logit contract')
                require(hashlib.sha256(a.tobytes()).hexdigest()==item['data_sha256'],'Model modified input')
                probabilities=logits.sigmoid().cpu().numpy()[0,0].copy();logits=logits.cpu().numpy()[0,0].copy()
                require(np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1)),'Probability contract')
                stem=f'{item["context"]}-{item["start"]}'
                np.save(out/(stem+'-logits.npy'),logits,allow_pickle=False);np.save(out/(stem+'-probabilities.npy'),probabilities,allow_pickle=False)
                records.append(dict(context=item['context'],start=item['start'],input=item,
                    logits=pin(out/(stem+'-logits.npy')),probabilities=pin(out/(stem+'-probabilities.npy')),
                    forward_seconds=time.monotonic()-t,score_min=float(probabilities.min()),score_max=float(probabilities.max())))
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'Model RSS exceeded')
                print(json.dumps(dict(completed_forwards=len(records),context=item['context'],window_start=item['start'])),flush=True)
        require(attempted==20 and state_digest(model)==digest and all(p.grad is None for p in model.parameters()),'Final model state/forward count')
        check_freeze('MODEL_GO.json')
        save(out/'COMPLETE.json',dict(status='PASS',GO=pin(HERE/'MODEL_GO.json'),records=records,attempted_forwards=attempted,
             completed_forwards=len(records),state_sha256=digest,model_state_unchanged=True,model_metadata=meta,
             torch_version=torch.__version__,numpy_version=np.__version__,threads=torch.get_num_threads(),
             elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             created_at_utc=now(),training_calls=0,network_calls=0))
    except Exception as e:
        save(out/'FAILURE.json',dict(status='STOP',attempted_forwards=attempted,completed_forwards=len(records),error=type(e).__name__+': '+str(e)))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','worker']);args=parser.parse_args()
    (prepare if args.stage=='prepare' else worker)()
