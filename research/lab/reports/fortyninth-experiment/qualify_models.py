"""Twenty fixed-tile forwards; no target map or contrast selection."""
from pathlib import Path
import sys,json,hashlib,importlib.util,gc,time,subprocess,resource
import numpy as np,zarr,torch,cv2
from prepare import pin,save
P=Path(__file__).resolve().parent;W=P/'work';R=P.parent;P46=R/'fortysixth-experiment';OLD=R/'fortyfifth-experiment'
assert json.loads((P/'RENDER_VERIFICATION.json').read_text())['status']=='PASS'
assert not (P/'INFERENCE_GO.json').exists()
prior=json.loads((P46/'INFERENCE_QUALIFICATION.json').read_text())
for row in prior['models']:
    r=pin(row['checkpoint']['path']);assert r['sha256']==row['checkpoint']['sha256'] and r['size_bytes']==row['checkpoint']['bytes']
S=OLD/'source-evidence/pherc343/ink/scripts';hecate_source=OLD/'source-evidence/hecate-hecate.py'
save(P/'INFERENCE_GO.json',dict(script=pin(__file__),render_verification=pin(P/'RENDER_VERIFICATION.json'),checkpoints=[pin(r['checkpoint']['path']) for r in prior['models']],hecate_source=pin(hecate_source),prep_source=pin(S/'hecate_prep.py'),centre_depth_d=0,villa_indices=list(range(41,24,-1)),hecate_centre=32.5,global_hecate_tile_yxhw=[1120,1120,64,64],tile_forwards=20))
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
# This canvas is a preparation device, not a claimed observed full volume.
native=np.load(W/'render66.npy');virtual=W/'qualification-virtual-unknown-zero.zarr'
g=zarr.open_group(str(virtual),mode='w',zarr_format=2);a=g.create_array('0',shape=(66,2580,2580),chunks=(66,128,128),dtype='u1');a[:,1216:1344,1216:1344]=native
g.attrs['unknown_outside_xywh']=[1216,1216,128,128];g.attrs['scope']='ONLY the fixed qualification tile is observed; zeros elsewhere are unknown placeholders.'
known=np.zeros((2580,2580),np.float32);known[1216:1344,1216:1344]=1
resized=cv2.resize(known,(2322,2322),interpolation=cv2.INTER_AREA);selected=np.s_[1120:1184,1120:1184]
assert np.array_equal(resized[selected],np.ones((64,64),np.float32))
# Independent placeholder perturbation at an actual source plane, same full grid.
zero=np.zeros_like(known);zero[1216:1344,1216:1344]=native[32]
other=zero.copy();other[known==0]=255
assert np.array_equal(cv2.resize(zero,(2322,2322),interpolation=cv2.INTER_AREA)[selected],cv2.resize(other,(2322,2322),interpolation=cv2.INTER_AREA)[selected])
subprocess.run([sys.executable,'-B',str(S/'hecate_prep.py'),'--src',str(virtual),'--out',str(W/'hecate-qualification-prep.zarr'),'--native-um','8.64','--native-um-z','8.64','--target-um','9.6','--z-center','32.5','--nz','16'],check=True)
del zero,other,known,resized;gc.collect()
sys.path.insert(0,str(P46/'runtime-source'));from vesuvius.ink_detection.inference import infer
infer.load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
infer.load_flat_inference_state=lambda m,s:m.load_state_dict(s,strict=True)
rows=[]
for old in prior['models']:
    name=old['model'];ck=Path(old['checkpoint']['path'])
    if name!='hecate':
        args=infer.parse_args([str(virtual),str(ck),str(W/'unused.tif'),'--no-compile','--layer-start','25','--layer-end','42','--direction','reverse','--batch-size','4','--num-workers','0'])
        conf=infer.configure_model(args);model=conf.model
        indices=infer.select_layer_indices(66,layer_start=25,layer_end=42,output_depth=17,direction='reverse');assert indices.tolist()==list(range(41,24,-1))
        reader=infer.FlatPatchReader(input_path=virtual,resolution='0',depth_axis_first=True,height=2580,width=2580,layer_indices=indices,output_depth=17,preprocessing=conf.preprocessing)
        raw=reader.read(1216,1216,128,128).transpose(2,0,1)
        norm=infer.normalize_flat_patch(raw,conf.preprocessing);x=torch.from_numpy(norm[None,None]);forward=lambda x:model(x)
        preprocessing=conf.preprocessing
    else:
        spec=importlib.util.spec_from_file_location('hecate',hecate_source);hec=importlib.util.module_from_spec(spec);spec.loader.exec_module(hec)
        model=hec.load_model(ck,'cpu');arr=zarr.open_group(str(W/'hecate-qualification-prep.zarr'),mode='r')['0']
        raw=np.asarray(arr[:,1120:1184,1120:1184])[::-1].copy();norm=raw.astype(np.float32)/model.divisor;x=torch.from_numpy(norm[None,None])
        support=torch.from_numpy(raw.any(0))[None,None,None].expand(1,1,16,64,64);forward=lambda x:model(x,support.repeat(x.shape[0],1,1,1,1));preprocessing='author interpolated9.6um then uint8, reversed, /255; raw-any support mask'
    np.save(W/f'qualify-{name}-raw.npy',raw);np.save(W/f'qualify-{name}-normalized.npy',norm)
    with torch.inference_mode():
        t=time.monotonic();single=forward(x).sigmoid();st=time.monotonic()-t
        t=time.monotonic();batch=forward(x.repeat(4,1,1,1,1)).sigmoid();bt=time.monotonic()-t
    err=float((batch-single).abs().max());finite=bool(torch.isfinite(batch).all() and torch.isfinite(single).all())
    np.save(W/f'qualify-{name}-single.npy',single.numpy());np.save(W/f'qualify-{name}-batch4.npy',batch.numpy())
    row=dict(model=name,checkpoint=pin(ck),strict_load=True,preprocessing=preprocessing,input_shape=list(raw.shape),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),single_seconds=st,batch4_seconds=bt,max_batch_difference=err,finite=finite,tile_forwards=5,status='PASS' if finite and err<1e-4 else 'FAIL')
    rows.append(row);save(W/f'qualification-{name}.json',row);print(json.dumps(row),flush=True)
    assert row['status']=='PASS'
    del model,x,single,batch,forward
    if name!='hecate':del conf,reader
    gc.collect()
save(P/'INFERENCE_QUALIFICATION.json',dict(status='PASS',models=rows,total_tile_forwards=20,full_target_maps=0,torch=torch.__version__,python=sys.version,threads=4,precision='fp32 CPU',hecate_known_support_fraction=1.0,unknown_placeholder_perturbation_invariant=True,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
