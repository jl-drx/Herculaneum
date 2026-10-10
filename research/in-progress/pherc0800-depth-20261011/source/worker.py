"""Fixed shifted-depth maps with original cohorts, depth-bound commits and replay."""
from pathlib import Path
import sys,json,hashlib,time,importlib.util,gc,os,datetime,fcntl
import numpy as np,torch,zarr,cv2
from common import *
PREV=R/'fiftyfirst-experiment';V=PREV/'work/model-inputs';O=W/'inference'
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
  ix=self.count;self.count+=1;assert ix<len(self.cohorts);entry=self.cohorts[ix];cohort=entry['cohort_yx'];assert args[0].shape[0]==len(cohort)
  meta=self.folder/f'batch-{ix:05}.json';arr=self.folder/f'batch-{ix:05}.npy';inputs=[digest_array(a.detach().cpu().numpy()) for a in args]
  if meta.exists():
   rec=json.loads(meta.read_text());assert rec['binding']==self.binding and rec['index']==ix and rec['original_batch_index']==entry['original_batch_index'] and rec['cohort_yx']==cohort and rec['inputs']==inputs
   assert pin(arr)==rec['logits'];out=torch.from_numpy(np.load(arr,allow_pickle=False));self.reused+=1
  else:
   assert not self.readonly,'Read-only replay lacks committed logits'
   with torch.inference_mode():out=self.base(*args)
   assert torch.isfinite(out).all();atomnpy(arr,out.detach().cpu().numpy());atomjson(meta,dict(binding=self.binding,index=ix,original_batch_index=entry['original_batch_index'],cohort_yx=cohort,inputs=inputs,logits=pin(arr)));self.new+=1
   if self.stop_after is not None and self.new==self.stop_after:raise InjectedStop()
  assert torch.isfinite(out).all()
  if self.progress:
   atomjson(O/'PROGRESS.json',dict(status='RUNNING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),job=self.binding['job'],stage='cached replay' if self.readonly else 'inference',committed_batches=self.count,expected_job_batches=len(self.cohorts),new_batches_this_pass=self.new,reused_batches_this_pass=self.reused,completed_maps=len(list(O.glob('*/COMPLETE.json'))),expected_maps=32,pass_seconds=time.monotonic()-self.started))
  return out
class SelectedVolume:
 def __init__(self,volume,known,allowed):self.v=volume;self.known=known;self.allowed=set(allowed);self.shape=volume.shape;self.dtype=volume.dtype
 def __getitem__(self,item):
  zs,ys,xs=item;y,x=ys.start,xs.start
  if (y,x) not in self.allowed:return np.zeros((16,ys.stop-y,xs.stop-x),np.uint8)
  assert self.known[ys,xs].all();return np.asarray(self.v[item])

def calculate(go,go_sha,name,phase,direction,depth,model,conf,grid,folder,stop_after=None,readonly=False,direct=False,progress=False):
 py,px=(0,0) if phase=='plain' else (40,200);H,WW=2580+py,2580+px;cx,cy,cw,ch=go['core_xywh'];sl=np.s_[cy+py:cy+py+ch,cx+px:cx+px+cw]
 cached=model if direct else CacheModel(model,folder,dict(go_sha256=go_sha,job=[name,phase,direction,depth],depth=depth,positions=next(v for v in go['depth_positions'] if v['d']==depth)),grid['batches'],stop_after,readonly,progress)
 if name!='hecate':
  indices=np.arange(25+depth,42+depth);indices=indices[::-1] if direction=='reverse' else indices
  reader=infer.FlatPatchReader(input_path=V/(phase+'.zarr'),resolution='0',depth_axis_first=True,height=H,width=WW,layer_indices=indices,output_depth=17,preprocessing=conf.preprocessing)
  assert all(len(b['cohort_yx'])==4 for b in grid['batches']);blocks=[infer.Block(y,x,128,128) for b in grid['batches'] for y,x in b['cohort_yx']];known=np.load(V/(phase+'-known.npy'))
  for b in blocks:assert known[b.y0:b.y0+b.valid_h,b.x0:b.x0+b.valid_w].all()
  dataset=infer.FlatBlockDataset(reader=reader,blocks=blocks,patch_size=128,preprocessing=conf.preprocessing);loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)
  weight=infer.compute_importance_map_2d(patch_size=(128,128),mode='hann').numpy();prob=np.zeros((H,WW),np.float32);mass=np.zeros_like(prob)
  acc=infer.ChunkAccumulator(shape=(H,WW),chunk_shape=(128,128),prob_sum_store=prob,weight_sum_store=mass,contribution_counts=infer.compute_chunk_contribution_counts(blocks,chunk_shape=(128,128)))
  infer.run_block_inference(loader=loader,model=cached,accumulator=acc,weight_map=weight,mask=None,device=torch.device('cpu'),amp_dtype=None,tta_axes=(),tta_batch_size=None);acc.flush_remaining()
  num=prob[sl].copy();den=mass[sl].copy();p=num.copy();np.divide(p,den,out=p,where=den>1e-6);np.clip(p,0,1,out=p)
  raw=(p*255).astype(np.uint8);aligned=raw.astype(np.float32)/255;extra={'numerator':num,'denominator':den,'unquantized':p}
 else:
  v=zarr.open_group(str(W/f'hecate-{phase}-d{depth:+03}.zarr'),mode='r')['0'];known=np.load(V/('hecate-'+phase+'-known.npy'));subset=SelectedVolume(v,known,[(y,x) for b in grid['batches'] for y,x in b['cohort_yx']]);raw=np.zeros(v.shape[-2:],np.uint8)
  hec.predict(cached,subset,raw,reverse=direction=='reverse',stride=32,batch_size=4,precision='fp32')
  aligned=cv2.resize(raw.astype(np.float32)/255,(WW,H),interpolation=cv2.INTER_LINEAR)[sl].copy();extra={}
 display=np.rint(aligned*255).astype(np.uint8);assert aligned.shape==(ch,cw) and np.isfinite(aligned).all() and (aligned>=0).all() and (aligned<=1).all()
 if not direct:assert cached.count==len(grid['batches'])
 counts=None if direct else dict(model_calls=cached.count,new_forwards=cached.new,reused_forwards=cached.reused,model_tiles=sum(len(b['cohort_yx']) for b in grid['batches']))
 return dict(raw_grid=raw,aligned=aligned,display=display,**extra),counts

def main():
 O.mkdir(exist_ok=True);lock=(O/'WORKER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (P/'INFERENCE_RESULTS.json').exists(),'Study already complete'
 go=json.loads((P/'INFERENCE_GO.json').read_text());go_sha=pin(P/'INFERENCE_GO.json')['sha256']
 for r in go['pins']+go['runtime_pins']+[m['checkpoint'] for m in go['models']]:assert pin(r['path'])==r,r['path']
 assert go['depths']==[-15,15] and go['expected_maps']==32 and go['expected_batches']==3096
 assert json.loads((P/'BATCH_QUALIFICATION.json').read_text())['status']=='PASS'
 configure();receipts=[];start_all=time.monotonic();support=np.load(W/'output-support.npz')
 for row in go['models']:
  name=row['name'];model,conf=load(row);family='hecate' if name=='hecate' else 'villa'
  for depth in go['depths']:
   for phase in ['plain','padded']:
    grid=next(g for g in go['grids'] if (g['family'],g['phase'])==(family,phase))
    for direction in ['forward','reverse']:
     job=f'{name}-{phase}-{direction}-d{depth:+03}';folder=O/job;folder.mkdir(exist_ok=True);done=folder/'COMPLETE.json'
     if done.exists():
      rec=json.loads(done.read_text());assert rec['go_sha256']==go_sha and [rec['model'],rec['phase'],rec['direction'],rec['depth']]==[name,phase,direction,depth]
      for r in rec['outputs'].values():assert pin(r['path'])==r
      receipts.append(rec);continue
     start=time.monotonic();arrays,counts=calculate(go,go_sha,name,phase,direction,depth,model,conf,grid,folder,progress=True)
     replay,rc=calculate(go,go_sha,name,phase,direction,depth,model,conf,grid,folder,readonly=True,progress=True)
     assert arrays.keys()==replay.keys() and all(np.array_equal(arrays[k],replay[k]) for k in arrays) and rc['new_forwards']==0
     outputs={}
     for k,a in arrays.items():path=folder/(k+'.npy');atomnpy(path,a);outputs[k]=pin(path)
     mask=support[family+'_'+phase];masked=arrays['aligned'].copy();masked[~mask]=np.nan;atomnpy(folder/'supported.npy',masked);outputs['supported']=pin(folder/'supported.npy')
     rec=dict(status='PASS',model=name,phase=phase,direction=direction,depth=depth,go_sha256=go_sha,core_xywh=go['core_xywh'],seconds=time.monotonic()-start,counts=counts,read_only_replay=dict(status='PASS',**rc),outputs=outputs,support_mask=pin(W/'output-support.npz'))
     atomjson(done,rec);receipts.append(rec);print('COMPLETED',job,json.dumps(counts),flush=True)
  del model,conf;gc.collect()
 assert len(receipts)==32 and sum(r['counts']['model_calls'] for r in receipts)==3096
 result=dict(status='PASS',maps=receipts,all_saved_logit_replays_exact=True,total_unique_model_batches=3096,total_model_tiles=sum(r['counts']['model_tiles'] for r in receipts),current_process_seconds=time.monotonic()-start_all,precision='fp32 CPU',batch_size=4,gpu_spend_GBP=0)
 atomjson(P/'INFERENCE_RESULTS.json',result);atomjson(O/'PROGRESS.json',dict(status='COMPLETE',completed_maps=32,expected_maps=32,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('ALL_MAPS_COMPLETE',flush=True)

if __name__=='__main__':main()
