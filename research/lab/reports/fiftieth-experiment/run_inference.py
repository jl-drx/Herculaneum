"""Fixed candidate16-map comparison with immutable logits and replayable accumulation."""
from pathlib import Path
import os,sys,json,hashlib,time,importlib.util,gc
import numpy as np,torch,zarr,cv2
P=Path(__file__).resolve().parent;R=P.parent;W=P/'work';V=W/'model-inputs';O=W/'inference';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'fortysixth-experiment/runtime-source'))
from vesuvius.ink_detection.inference import infer
from admit_candidate import pin,save

def atomjson(path,value):
 tmp=path.with_name(path.name+'.partial')
 with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def atomnpy(path,a):
 tmp=path.with_name(path.name+'.partial')
 with tmp.open('wb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def digest_array(a):
 a=np.asarray(a);return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
class InjectedStop(Exception):pass
class CacheModel(torch.nn.Module):
 def __init__(self,model,folder,binding,stop_after=None,readonly=False):
  super().__init__();self.base=model;self.folder=folder;self.binding=binding;self.count=0;self.new=0;self.reused=0;self.stop_after=stop_after;self.readonly=readonly
  for name in ['patch_size','divisor','margin']:
   if hasattr(model,name):setattr(self,name,getattr(model,name))
 def forward(self,*args):
  ix=self.count;self.count+=1;meta=self.folder/f'tile-{ix:03}.json';arr=self.folder/f'tile-{ix:03}.npy';inputs=[digest_array(a.detach().cpu().numpy()) for a in args]
  if meta.exists():
   rec=json.loads(meta.read_text());assert rec['binding']==self.binding and rec['inputs']==inputs
   assert pin(arr)==rec['logits'];out=torch.from_numpy(np.load(arr,allow_pickle=False));self.reused+=1
  else:
   assert not self.readonly,'Read-only replay lacks committed logits'
   # Uncommitted .npy can be safely replaced; without its receipt it is not reused.
   with torch.inference_mode():out=self.base(*args)
   assert torch.isfinite(out).all();atomnpy(arr,out.detach().cpu().numpy())
   atomjson(meta,dict(binding=self.binding,index=ix,inputs=inputs,logits=pin(arr)));self.new+=1
   if self.stop_after is not None and self.new==self.stop_after:raise InjectedStop()
  assert torch.isfinite(out).all();return out
class SelectedVolume:
 def __init__(self,volume,known,allowed):self.v=volume;self.known=known;self.allowed=set(allowed);self.shape=volume.shape;self.dtype=volume.dtype
 def __getitem__(self,item):
  zs,ys,xs=item;y,x=ys.start,xs.start
  if (y,x) not in self.allowed:return np.zeros((16,ys.stop-y,xs.stop-x),np.uint8)
  assert self.known[ys,xs].all();return np.asarray(self.v[item])

def calculate(name,phase,direction,model,conf,grid,folder,stop_after=None,readonly=False):
 py,px=(0,0) if phase=='plain' else (40,200);H,WW=2300+py,2260+px;cx,cy,cw,ch=go['core_xywh'];sl=np.s_[cy+py:cy+py+ch,cx+px:cx+px+cw]
 cached=CacheModel(model,folder,dict(go_sha256=go_sha,job=[name,phase,direction]),stop_after,readonly)
 if name!='hecate':
  indices=np.arange(25,42);indices=indices[::-1] if direction=='reverse' else indices
  reader=infer.FlatPatchReader(input_path=V/(phase+'.zarr'),resolution='0',depth_axis_first=True,height=H,width=WW,layer_indices=indices,output_depth=17,preprocessing=conf.preprocessing)
  blocks=[infer.Block(y,x,h,w) for x,y,w,h in grid['patches_xywh']]
  known=np.load(V/(phase+'-known.npy'))
  for b in blocks:assert known[b.y0:b.y0+b.valid_h,b.x0:b.x0+b.valid_w].all()
  dataset=infer.FlatBlockDataset(reader=reader,blocks=blocks,patch_size=128,preprocessing=conf.preprocessing)
  loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
  weight=infer.compute_importance_map_2d(patch_size=(128,128),mode='hann').numpy();prob=np.zeros((H,WW),np.float32);mass=np.zeros_like(prob)
  acc=infer.ChunkAccumulator(shape=(H,WW),chunk_shape=(128,128),prob_sum_store=prob,weight_sum_store=mass,contribution_counts=infer.compute_chunk_contribution_counts(blocks,chunk_shape=(128,128)))
  infer.run_block_inference(loader=loader,model=cached,accumulator=acc,weight_map=weight,mask=None,device=torch.device('cpu'),amp_dtype=None,tta_axes=(),tta_batch_size=None);acc.flush_remaining()
  num=prob[sl].copy();den=mass[sl].copy();p=num.copy();np.divide(p,den,out=p,where=den>1e-6);np.clip(p,0,1,out=p)
  raw=(p*255).astype(np.uint8);aligned=raw.astype(np.float32)/255;extra={'numerator':num,'denominator':den,'unquantized':p}
 else:
  v=zarr.open_group(str(V/('hecate-'+phase+'.zarr')),mode='r')['0'];known=np.load(V/('hecate-'+phase+'-known.npy'))
  subset=SelectedVolume(v,known,[(y,x) for x,y,w,h in grid['patches_xywh']]);raw=np.zeros(v.shape[-2:],np.uint8)
  hec.predict(cached,subset,raw,reverse=direction=='reverse',stride=32,batch_size=1,precision='fp32')
  aligned=cv2.resize(raw.astype(np.float32)/255,(WW,H),interpolation=cv2.INTER_LINEAR)[sl].copy();extra={}
 display=np.rint(aligned*255).astype(np.uint8)
 assert aligned.shape==(128,128) and np.isfinite(aligned).all() and (aligned>=0).all() and (aligned<=1).all()
 return dict(raw_grid=raw,aligned=aligned,display=display,**extra),dict(model_calls=cached.count,new_forwards=cached.new,reused_forwards=cached.reused)

def main():
 global go,go_sha,hec
 go=json.loads((P/'INFERENCE_GO.json').read_text());go_sha=pin(P/'INFERENCE_GO.json')['sha256']
 for r in go['pins']+go['runtime_pins']+[r['checkpoint'] for r in go['models']]:assert pin(r['path'])==r,r['path']
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 infer.load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
 infer.load_flat_inference_state=lambda m,s:m.load_state_dict(s,strict=True)
 spec=importlib.util.spec_from_file_location('hecate_original',R/'fortyfifth-experiment/source-evidence/hecate-hecate.py');hec=importlib.util.module_from_spec(spec);spec.loader.exec_module(hec)
 receipts=[];allstart=time.monotonic()
 for row in go['models']:
  name=row['name'];ck=Path(row['checkpoint']['path'])
  if name=='hecate':model=hec.load_model(ck,'cpu');conf=None
  else:
   args=infer.parse_args(['unused.zarr',str(ck),'unused.tif','--no-compile']);conf=infer.configure_model(args);model=conf.model
   assert conf.patch_size==128 and conf.input_depth==17
  for phase in ['plain','padded']:
   grid=next(g for g in go['grids'] if g['phase']==phase and g['family']==('hecate' if name=='hecate' else 'villa'))
   for direction in go['directions']:
    job=name+'-'+phase+'-'+direction;folder=O/job;folder.mkdir(exist_ok=True);receipt=folder/'COMPLETE.json'
    if receipt.exists():
     rec=json.loads(receipt.read_text());assert rec['go_sha256']==go_sha
     for r in rec['outputs'].values():assert pin(r['path'])==r
     receipts.append(rec);continue
    start=time.monotonic();injected=0
    if name=='dnative' and phase=='plain' and direction=='forward' and not (O/'INJECTED_STOP.json').exists():
     try:calculate(name,phase,direction,model,conf,grid,folder,stop_after=2)
     except InjectedStop:
      injected=2;save(O/'INJECTED_STOP.json',dict(status='EXPECTED_STOP_AFTER_TWO_COMMITTED_FORWARDS',job=job,go_sha256=go_sha,completed_map=False,committed_receipts=2))
     else:raise AssertionError('Fault injection did not stop')
    arrays,counts=calculate(name,phase,direction,model,conf,grid,folder)
    replay,replaycounts=calculate(name,phase,direction,model,conf,grid,folder,readonly=True)
    assert arrays.keys()==replay.keys() and all(np.array_equal(arrays[k],replay[k]) for k in arrays)
    assert replaycounts['new_forwards']==0 and replaycounts['reused_forwards']==counts['model_calls']
    outputs={}
    for key,a in arrays.items():f=folder/(key+'.npy');atomnpy(f,a);outputs[key]=pin(f)
    rec=dict(status='PASS',model=name,phase=phase,direction=direction,go_sha256=go_sha,core_xywh=go['core_xywh'],seconds=time.monotonic()-start,counts=counts,initial_fault_injection_forwards=injected,read_only_replay= dict(status='PASS',**replaycounts),outputs=outputs)
    save(receipt,rec);receipts.append(rec)
    atomjson(O/'PROGRESS.json',dict(completed_maps=len(receipts),expected_maps=16,last_completed=job,seconds=time.monotonic()-allstart))
    print('COMPLETED',job,json.dumps(counts),flush=True)
  del model,conf;gc.collect()
 assert len(receipts)==16
 result=dict(status='PASS',maps=receipts,total_unique_model_forwards=sum(r['counts']['model_calls'] for r in receipts),planned_maximum_forwards=288,all_saved_logit_replays_exact=True,seconds=time.monotonic()-allstart,precision='fp32 CPU',batch_size=1,gpu_spend_GBP=0)
 save(P/'INFERENCE_RESULTS.json',result);print('ALL_MAPS_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='maps'}),flush=True)
if __name__=='__main__':main()
