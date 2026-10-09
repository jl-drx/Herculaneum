"""Offline synthetic strict-load/forward check; no papyrus accuracy claim."""
from pathlib import Path
import gc,hashlib,importlib.util,json,sys,time,resource
P=Path(__file__).resolve().parent;W=P/'work';OLD=P.parent/'fortyfifth-experiment';sys.path.insert(0,str(P/'runtime-source'))
import numpy as np,torch
from vesuvius.ink_detection.inference import infer
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
infer.load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
infer.load_flat_inference_state=lambda m,s:m.load_state_dict(s,strict=True)
models=[('dnative',W/'dense_native-016000.rebuilt.pth'),('seed42',P.parents[1]/'data/models/ink9um-seed42-step075000.pth'),('seed43',P.parents[1]/'data/models/ink9um-seed43-step075000.pth'),('hecate',W/'hecate_9.6um.pth')]
spec=importlib.util.spec_from_file_location('hecate',OLD/'source-evidence/hecate-hecate.py');hec=importlib.util.module_from_spec(spec);spec.loader.exec_module(hec)
rows=[]
for name,checkpoint in models:
    if name=='hecate':
        model=hec.load_model(checkpoint,'cpu');raw=np.random.default_rng(20261009).integers(1,256,size=(16,64,64),dtype=np.uint8);norm=raw.astype(np.float32)/255
    else:
        args=infer.parse_args(['unused.zarr',str(checkpoint),'unused.tif','--no-compile']);conf=infer.configure_model(args);model=conf.model
        raw=np.random.default_rng(20261009).integers(1,256,size=(17,128,128),dtype=np.uint8);norm=infer.normalize_flat_patch(raw,conf.preprocessing)
    x=torch.from_numpy(norm[None,None]);times=[]
    with torch.inference_mode():
        t=time.monotonic();a=model(x).sigmoid();times.append(time.monotonic()-t)
        t=time.monotonic();b=model(x.repeat(4,1,1,1,1)).sigmoid();times.append(time.monotonic()-t)
    err=float((a-b[:1]).abs().max());assert torch.isfinite(b).all() and err<1e-4,(name,err)
    np.save(W/f'synthetic-{name}-single.npy',a.numpy());np.save(W/f'synthetic-{name}-batch.npy',b.numpy())
    with checkpoint.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    rows.append(dict(model=name,checkpoint=str(checkpoint),checkpoint_sha256=sha,strict_load=True,input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),single_seconds=times[0],batch4_seconds=times[1],batch_probability_max_difference=err,output_shape=list(a.shape),output_range=[float(b.min()),float(b.max())],tile_forwards=5))
    print(json.dumps(rows[-1]),flush=True);del model,a,b,x;gc.collect()
result=dict(status='PASS',kind='synthetic runtime qualification, not CT inference',python=sys.version,torch=torch.__version__,cpu_threads=4,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,total_tile_forwards=20,models=rows)
(P/'RUNTIME_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n')
