"""Prepare exact author grids; qualify strict CPU loading and batching on fixed tiles."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,subprocess,sys,time
P=Path(__file__).resolve().parent;W=P/'work';OLD=P.parent/'fortyfifth-experiment';NAME='concat_w047-w048_R5B2_z9500-11000'
sys.path.insert(0,str(P/'runtime-source'))
import numpy as np,torch,zarr
from vesuvius.ink_detection.inference import infer
S=OLD/'source-evidence/pherc343/ink/scripts'
assert json.loads((P/'RENDER_VERIFICATION.json').read_text())['status']=='PASS'
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def pin(path):
    path=Path(path)
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha)
plain=W/f'{NAME}_66.zarr';padded=W/f'{NAME}_66_padded.zarr'
subprocess.run([sys.executable,'-B',str(S/'pad_render.py'),str(plain),str(padded),'--pad-rows','40','--pad-cols','200'],check=True)
for phase,path in [('plain',plain),('padded',padded)]:
    view=W/f'{phase}-level0.zarr';view.mkdir();(view/'.zgroup').write_text('{"zarr_format":2}\n');(view/'0').symlink_to(path/'0',target_is_directory=True)
    subprocess.run([sys.executable,'-B',str(S/'hecate_prep.py'),'--src',str(path),'--out',str(W/f'hecate-{phase}.zarr'),'--native-um','8.64','--native-um-z','8.64','--target-um','9.6','--z-center','33.5','--nz','16'],check=True)
models=[('dnative',W/'dense_native-016000.rebuilt.pth',None),('hybrid_3d2d-seed42',P.parents[1]/'data/models/ink9um-seed42-step075000.pth','e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab'),('hybrid_3d2d-seed43',P.parents[1]/'data/models/ink9um-seed43-step075000.pth','2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f')]
inf_orig=infer.load_flat_inference_state
infer.load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
infer.load_flat_inference_state=lambda m,s:m.load_state_dict(s,strict=True)
rows=[]
for name,ck,expected in models:
    cp=pin(ck)
    if expected:assert cp['sha256']==expected
    args=infer.parse_args([str(W/'plain-level0.zarr'),str(ck),str(W/'unused.tif'),'--no-compile','--layer-start','26','--layer-end','43','--direction','reverse','--batch-size','4','--num-workers','0'])
    conf=infer.configure_model(args);root=zarr.open_group(str(W/'plain-level0.zarr'),mode='r')
    layers=infer.select_layer_indices(66,layer_start=26,layer_end=43,output_depth=17,direction='reverse');assert layers.tolist()==list(range(42,25,-1))
    reader=infer.FlatPatchReader(input_path=W/'plain-level0.zarr',resolution='0',depth_axis_first=True,height=2560,width=6720,layer_indices=layers,output_depth=17,preprocessing=conf.preprocessing)
    raw=reader.read(1024,3072,128,128).transpose(2,0,1);assert raw.shape==(17,128,128)
    norm=infer.normalize_flat_patch(raw,conf.preprocessing);x=torch.from_numpy(norm[None,None])
    with torch.inference_mode():
        t=time.monotonic();a=conf.model(x).sigmoid();single=time.monotonic()-t
        t=time.monotonic();b=conf.model(x.repeat(4,1,1,1,1)).sigmoid();batch=time.monotonic()-t
    err=float((a-b[:1]).abs().max());assert torch.isfinite(b).all() and err<1e-4,(name,err)
    counts={}
    for phase in ['plain','padded']:
        group=zarr.open_group(str(W/f'{phase}-level0.zarr'),mode='r');h,w=group['0'].shape[-2:]
        occ,scale,level=infer.build_lowres_block_mask(group,height=h,width=w,user_mask=None);counts[phase]=len(infer.iter_blocks((h,w),128,64,occ,scale))
    np.save(W/f'qualify-{name}-single.npy',a.numpy());np.save(W/f'qualify-{name}-batch.npy',b.numpy())
    rows.append(dict(model=name,checkpoint=cp,strict_load=True,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),single_seconds=single,batch4_seconds=batch,probability_max_batch_difference=err,patches=counts,estimated_seconds=sum(counts.values())*batch/4,new_tile_forwards=5,precision='fp32 CPU; published Villa CUDA auto AMP differs'))
    print(json.dumps(rows[-1]),flush=True);del conf,a,b,x;gc.collect()
spec=importlib.util.spec_from_file_location('hecate',OLD/'source-evidence/hecate-hecate.py');hec=importlib.util.module_from_spec(spec);spec.loader.exec_module(hec)
model=hec.load_model(W/'hecate_9.6um.pth','cpu');v=zarr.open_group(str(W/'hecate-plain.zarr'),mode='r')['0']
# Fixed global Hecate tile nearest the predefined native debugging origin.
y,x0=928,2752;raw=np.asarray(v[:,y:y+64,x0:x0+64])[::-1].copy();norm=raw.astype(np.float32)/model.divisor;x=torch.from_numpy(norm[None,None]);support=torch.from_numpy(raw.any(0))[None,None,None].expand(1,1,16,64,64)
with torch.inference_mode():
    t=time.monotonic();a=model(x,support).sigmoid();single=time.monotonic()-t
    t=time.monotonic();b=model(x.repeat(4,1,1,1,1),support.repeat(4,1,1,1,1)).sigmoid();batch=time.monotonic()-t
err=float((a-b[:1]).abs().max());assert torch.isfinite(b).all() and err<1e-4,err
np.save(W/'qualify-hecate-single.npy',a.numpy());np.save(W/'qualify-hecate-batch.npy',b.numpy())
counts={}
for phase in ['plain','padded']:
    v=zarr.open_group(str(W/f'hecate-{phase}.zarr'),mode='r')['0'];h,w=v.shape[-2:];counts[phase]=len(hec.axis_starts(h,64,32))*len(hec.axis_starts(w,64,32))
rows.append(dict(model='hecate',checkpoint=pin(W/'hecate_9.6um.pth'),strict_load=True,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),single_seconds=single,batch4_seconds=batch,probability_max_batch_difference=err,patches=counts,estimated_seconds=sum(counts.values())*batch/4,new_tile_forwards=5,precision='fp32 CPU'))
print(json.dumps(rows[-1]),flush=True)
(P/'INFERENCE_QUALIFICATION.json').write_text(json.dumps(dict(status='PASS',torch=torch.__version__,python=sys.version,threads=4,models=rows,total_tile_forwards=20,total_estimated_seconds=sum(r['estimated_seconds'] for r in rows)),indent=2)+'\n')

