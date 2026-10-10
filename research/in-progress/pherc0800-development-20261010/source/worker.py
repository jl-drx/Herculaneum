"""Original-framework fixed maps, with hashed batch commits and exact replay."""
from pathlib import Path
import sys,json,hashlib,time,importlib.util,gc,os,datetime,fcntl
import numpy as np,torch,zarr,cv2
from common import *
V=W/'model-inputs';O=W/'inference'
sys.path.insert(0,str(R/'fortysixth-experiment/runtime-source'))
from vesuvius.ink_detection.inference import infer
spec=importlib.util.spec_from_file_location('hecate_original',R/'fortyfifth-experiment/source-evidence/hecate-hecate.py');hec=importlib.util.module_from_spec(spec);spec.loader.exec_module(hec)
def configure():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 infer.load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
 infer.load_flat_inference_state=lambda m,s:m.load_state_dict(s,strict=True)
def load(row):
 ck=Path(row['checkpoint']['path'])
 if row['name']=='hecate':return hec.load_model(ck,'cpu'),None
 args=infer.parse_args(['unused.zarr',str(ck),'unused.tif','--no-compile']);conf=infer.configure_model(args);assert conf.patch_size==128 and conf.input_depth==17
 return conf.model,conf
def digest_array(a):
 a=np.asarray(a);return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
class InjectedStop(Exception):pass
class CacheModel(torch.nn.Module):
 def __init__(self,model,folder,binding,cohorts,stop_after=None,readonly=False,progress=False):
  super().__init__();self.base=model;self.folder=folder;self.binding=binding;self.cohorts=cohorts;self.count=0;self.new=0;self.reused=0;self.stop_after=stop_after;self.readonly=readonly;self.progress=progress;self.started=time.monotonic();folder.mkdir(parents=True,exist_ok=True)
  for name in ['patch_size','divisor','margin']:
   if hasattr(model,name):setattr(self,name,getattr(model,name))
 def forward(self,*args):
  ix=self.count;self.count+=1;assert ix<len(self.cohorts);cohort=self.cohorts[ix];assert args[0].shape[0]==len(cohort)
  meta=self.folder/f'batch-{ix:05}.json';arr=self.folder/f'batch-{ix:05}.npy';inputs=[digest_array(a.detach().cpu().numpy()) for a in args]
  if meta.exists():
   rec=json.loads(meta.read_text());assert rec['binding']==self.binding and rec['index']==ix and rec['cohort_yx']==cohort and rec['inputs']==inputs
   assert pin(arr)==rec['logits'];out=torch.from_numpy(np.load(arr,allow_pickle=False));self.reused+=1
  else:
   assert not self.readonly,'Read-only replay lacks committed logits'
   with torch.inference_mode():out=self.base(*args)
   assert torch.isfinite(out).all();atomnpy(arr,out.detach().cpu().numpy());atomjson(meta,dict(binding=self.binding,index=ix,cohort_yx=cohort,inputs=inputs,logits=pin(arr)));self.new+=1
   if self.stop_after is not None and self.new==self.stop_after:raise InjectedStop()
  assert torch.isfinite(out).all()
  if self.progress:
   atomjson(O/'PROGRESS.json',dict(status='RUNNING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),job=self.binding['job'],stage='cached replay' if self.readonly else 'inference',committed_batches=self.count,expected_job_batches=len(self.cohorts),new_batches_this_pass=self.new,reused_batches_this_pass=self.reused,completed_maps=len(list(O.glob('*/COMPLETE.json'))),expected_maps=16,pass_seconds=time.monotonic()-self.started))
  return out
class SelectedVolume:
 def __init__(self,volume,known,allowed):self.v=volume;self.known=known;self.allowed=set(allowed);self.shape=volume.shape;self.dtype=volume.dtype
 def __getitem__(self,item):
  zs,ys,xs=item;y,x=ys.start,xs.start
  if (y,x) not in self.allowed:return np.zeros((16,ys.stop-y,xs.stop-x),np.uint8)
  assert self.known[ys,xs].all();return np.asarray(self.v[item])
def cohorts(family,phase,grid):
 path=V/(('hecate-' if family=='hecate' else '')+phase+'.zarr');v=zarr.open_group(str(path),mode='r')['0'];p=64 if family=='hecate' else 128;selected={(y,x) for x,y,w,h in grid['patches_xywh']}
 if family=='hecate':origins=[(y,x) for y in hec.axis_starts(v.shape[1],64,32) for x in hec.axis_starts(v.shape[2],64,32)];zs=slice(None)
 else:origins=[(y,x) for x,y,w,h in grid['patches_xywh']];zs=slice(25,42)
 batches=[]
 for i in range(0,len(origins),4):
  keep=[[y,x] for y,x in origins[i:i+4] if (y,x) in selected and np.asarray(v[zs,y:y+p,x:x+p]).any()]
  if keep:batches.append(keep)
 return batches

def calculate(go,go_sha,name,phase,direction,model,conf,grid,folder,stop_after=None,readonly=False,direct=False,progress=False):
 py,px=(0,0) if phase=='plain' else (40,200);H,WW=2580+py,2580+px;cx,cy,cw,ch=go['core_xywh'];sl=np.s_[cy+py:cy+py+ch,cx+px:cx+px+cw]
 cached=model if direct else CacheModel(model,folder,dict(go_sha256=go_sha,job=[name,phase,direction]),grid['cohorts_yx'],stop_after,readonly,progress)
 if name!='hecate':
  indices=np.arange(25,42);indices=indices[::-1] if direction=='reverse' else indices
  reader=infer.FlatPatchReader(input_path=V/(phase+'.zarr'),resolution='0',depth_axis_first=True,height=H,width=WW,layer_indices=indices,output_depth=17,preprocessing=conf.preprocessing)
  blocks=[infer.Block(y,x,h,w) for x,y,w,h in grid['patches_xywh']];known=np.load(V/(phase+'-known.npy'))
  for b in blocks:assert known[b.y0:b.y0+b.valid_h,b.x0:b.x0+b.valid_w].all()
  dataset=infer.FlatBlockDataset(reader=reader,blocks=blocks,patch_size=128,preprocessing=conf.preprocessing);loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)
  weight=infer.compute_importance_map_2d(patch_size=(128,128),mode='hann').numpy();prob=np.zeros((H,WW),np.float32);mass=np.zeros_like(prob)
  acc=infer.ChunkAccumulator(shape=(H,WW),chunk_shape=(128,128),prob_sum_store=prob,weight_sum_store=mass,contribution_counts=infer.compute_chunk_contribution_counts(blocks,chunk_shape=(128,128)))
  infer.run_block_inference(loader=loader,model=cached,accumulator=acc,weight_map=weight,mask=None,device=torch.device('cpu'),amp_dtype=None,tta_axes=(),tta_batch_size=None);acc.flush_remaining()
  num=prob[sl].copy();den=mass[sl].copy();p=num.copy();np.divide(p,den,out=p,where=den>1e-6);np.clip(p,0,1,out=p)
  raw=(p*255).astype(np.uint8);aligned=raw.astype(np.float32)/255;extra={'numerator':num,'denominator':den,'unquantized':p}
 else:
  v=zarr.open_group(str(V/('hecate-'+phase+'.zarr')),mode='r')['0'];known=np.load(V/('hecate-'+phase+'-known.npy'));subset=SelectedVolume(v,known,[(y,x) for x,y,w,h in grid['patches_xywh']]);raw=np.zeros(v.shape[-2:],np.uint8)
  hec.predict(cached,subset,raw,reverse=direction=='reverse',stride=32,batch_size=4,precision='fp32')
  aligned=cv2.resize(raw.astype(np.float32)/255,(WW,H),interpolation=cv2.INTER_LINEAR)[sl].copy();extra={}
 display=np.rint(aligned*255).astype(np.uint8);assert aligned.shape==(ch,cw) and np.isfinite(aligned).all() and (aligned>=0).all() and (aligned<=1).all()
 if not direct:assert cached.count==len(grid['cohorts_yx'])
 counts=None if direct else dict(model_calls=cached.count,new_forwards=cached.new,reused_forwards=cached.reused,model_tiles=sum(map(len,grid['cohorts_yx'])))
 return dict(raw_grid=raw,aligned=aligned,display=display,**extra),counts

def main():
 O.mkdir(exist_ok=True);lock=(O/'WORKER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (P/'INFERENCE_RESULTS.json').exists(),'Study already complete'
 go=json.loads((P/'INFERENCE_GO.json').read_text());go_sha=pin(P/'INFERENCE_GO.json')['sha256']
 for r in go['pins']+go['runtime_pins']+[r['checkpoint'] for r in go['models']]:assert pin(r['path'])==r,r['path']
 assert json.loads((P/'BATCH_QUALIFICATION.json').read_text())['status']=='PASS';configure();receipts=[];start_all=time.monotonic()
 for row in go['models']:
  name=row['name'];model,conf=load(row)
  for phase in ['plain','padded']:
   grid=next(g for g in go['grids'] if g['phase']==phase and g['family']==('hecate' if name=='hecate' else 'villa'))
   for direction in ['forward','reverse']:
    job=name+'-'+phase+'-'+direction;folder=O/job;folder.mkdir(exist_ok=True);done=folder/'COMPLETE.json'
    if done.exists():
     rec=json.loads(done.read_text());assert rec['go_sha256']==go_sha
     for r in rec['outputs'].values():assert pin(r['path'])==r
     receipts.append(rec);continue
    start=time.monotonic();arrays,counts=calculate(go,go_sha,name,phase,direction,model,conf,grid,folder,progress=True)
    replay,rc=calculate(go,go_sha,name,phase,direction,model,conf,grid,folder,readonly=True,progress=True)
    assert arrays.keys()==replay.keys() and all(np.array_equal(arrays[k],replay[k]) for k in arrays);assert rc['new_forwards']==0
    outputs={}
    for k,a in arrays.items():path=folder/(k+'.npy');atomnpy(path,a);outputs[k]=pin(path)
    # Raw diagnostic pixels outside masks are not admitted probability evidence.
    support=np.load(W/'output-support.npz');family='hecate' if name=='hecate' else 'villa';mask=support[family+'_'+phase];masked=arrays['aligned'].copy();masked[~mask]=np.nan;atomnpy(folder/'supported.npy',masked);outputs['supported']=pin(folder/'supported.npy')
    rec=dict(status='PASS',model=name,phase=phase,direction=direction,go_sha256=go_sha,core_xywh=go['core_xywh'],seconds=time.monotonic()-start,counts=counts,read_only_replay=dict(status='PASS',**rc),outputs=outputs,support_mask=pin(W/'output-support.npz'))
    atomjson(done,rec);receipts.append(rec);print('COMPLETED',job,json.dumps(counts),flush=True)
  del model,conf;gc.collect()
 assert len(receipts)==16;result=dict(status='PASS',maps=receipts,all_saved_logit_replays_exact=True,total_unique_model_batches=sum(r['counts']['model_calls'] for r in receipts),total_model_tiles=sum(r['counts']['model_tiles'] for r in receipts),current_process_seconds=time.monotonic()-start_all,precision='fp32 CPU',batch_size=4,gpu_spend_GBP=0)
 atomjson(P/'INFERENCE_RESULTS.json',result);atomjson(O/'PROGRESS.json',dict(status='COMPLETE',completed_maps=16,expected_maps=16,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('ALL_MAPS_COMPLETE',flush=True)
if __name__=='__main__':main()
