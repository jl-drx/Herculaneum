"""Predeclared shifted cases, exact worker baseline, restart and cache rejection."""
import copy,gc,json,time
import numpy as np,torch,zarr
from common import *
import worker as wk

class Counted(torch.nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.calls=0;self.tiles=0
  for name in ['patch_size','divisor','margin']:
   if hasattr(base,name):setattr(self,name,getattr(base,name))
 def forward(self,*args):
  self.calls+=1;self.tiles+=args[0].shape[0]
  return self.base(*args)

class PriorCache(Counted):
 def __init__(self,base,row,grid):
  super().__init__(base);self.row=row;self.grid=grid;self.original=Path(row['outputs']['aligned']['path']).parent
 def forward(self,*args):
  batch=self.grid['batches'][self.calls];self.calls+=1
  r=json.loads((self.original/f"batch-{batch['original_batch_index']:05}.json").read_text())
  assert r['binding']==dict(go_sha256='bd10e35e367fa982a3db8bd14be581766df9e06b7629186e3dba54f318cfc05e',job=[self.row['model'],self.row['phase'],self.row['direction']])
  assert r['cohort_yx']==batch['cohort_yx'] and r['inputs']==[wk.digest_array(a.numpy()) for a in args]
  assert pin(r['logits']['path'])==r['logits']
  return torch.from_numpy(np.load(r['logits']['path']))

def main():
 Q=W/'execution-qualification';Q.mkdir();go=json.loads((P/'INPUT_GO.json').read_text());sha=pin(P/'INPUT_GO.json')['sha256']
 for r in go['pins']+go['runtime_pins']+[m['checkpoint'] for m in go['models']]:assert pin(r['path'])==r,r['path']
 wk.configure();rows=[];baselines=[];counts=[];injection=None;fixture=None
 old=json.loads((wk.PREV/'INFERENCE_RESULTS.json').read_text());cx,cy,cw,ch=go['core_xywh'];crop=np.s_[cy-512:cy-512+ch,cx-512:cx-512+cw]
 for row in go['models']:
  name=row['name'];base,conf=wk.load(row);model=Counted(base);family='hecate' if name=='hecate' else 'villa'
  # Exercise the new worker itself on every existing d0 map, without forwards.
  for phase in ['plain','padded']:
   full=next(g for g in go['grids'] if (g['family'],g['phase'])==(family,phase))
   for direction in ['forward','reverse']:
    oldrow=next(r for r in old['maps'] if (r['model'],r['phase'],r['direction'])==(name,phase,direction))
    cached=PriorCache(model,oldrow,full)
    actual,_=wk.calculate(go,sha,name,phase,direction,0,cached,conf,full,Q/'unused',direct=True)
    for key in ['aligned','display']:
     rec=oldrow['outputs'][key];assert pin(rec['path'])==rec
     assert np.array_equal(actual[key],np.load(rec['path'],mmap_mode='r')[crop]),(name,phase,direction,key)
    assert cached.calls==len(full['batches']) and model.calls==0
    baselines.append(dict(model=name,phase=phase,direction=direction,status='PASS',batches=cached.calls,values=ch*cw,new_forwards=0))
    del cached,actual
  for depth in [-15,15]:
   for phase in ['plain','padded']:
    full=next(g for g in go['grids'] if (g['family'],g['phase'])==(family,phase));four=[b for b in full['batches'] if len(b['cohort_yx'])==4]
    grid=copy.deepcopy(full);grid['batches']=[four[0],four[-1]];assert four[0]['original_batch_index']!=four[-1]['original_batch_index']
    for direction in ['forward','reverse']:
     folder=Q/f'{name}-{phase}-{direction}-d{depth:+03}';folder.mkdir();tiles=grid['batches'][0]['cohort_yx'];before=model.calls
     if family=='villa':
      py,px=(0,0) if phase=='plain' else (40,200);indices=np.arange(25+depth,42+depth);indices=indices[::-1] if direction=='reverse' else indices
      reader=wk.infer.FlatPatchReader(input_path=wk.V/(phase+'.zarr'),resolution='0',depth_axis_first=True,height=2580+py,width=2580+px,layer_indices=indices,output_depth=17,preprocessing=conf.preprocessing)
      blocks=[wk.infer.Block(y,x,128,128) for y,x in tiles];ds=wk.infer.FlatBlockDataset(reader=reader,blocks=blocks,patch_size=128,preprocessing=conf.preprocessing);items=[ds[i] for i in range(4)];assert all(m[4] for a,m in items);args=[torch.stack([a for a,m in items])]
      raw=np.stack([reader.read(y,x,128,128).transpose(2,0,1) for y,x in tiles]);manual=np.stack([wk.infer.normalize_flat_patch(a,conf.preprocessing) for a in raw]);assert np.array_equal(args[0].numpy()[:,0],manual)
      probability=lambda a:wk.infer.logits_to_probabilities(a,image_hw=(128,128))
     else:
      v=zarr.open_group(str(W/f'hecate-{phase}-d{depth:+03}.zarr'),mode='r')['0'];raw=np.stack([np.asarray(v[:,y:y+64,x:x+64]) for y,x in tiles]);raw=raw[:,::-1].copy() if direction=='reverse' else raw
      image=torch.from_numpy((raw.astype(np.float32)/model.divisor)[:,None]);support=torch.from_numpy(raw).any(1)[:,None,None].expand(-1,1,16,-1,-1);args=[image,support];probability=lambda a:a.float().sigmoid()
     np.save(folder/'raw.npy',raw)
     with torch.inference_mode():
      t=time.monotonic();batch=probability(model(*args));seconds=time.monotonic()-t
      single=torch.cat([probability(model(*(a[i:i+1] for a in args))) for i in range(4)])
     difference=float((batch-single).abs().max());assert torch.isfinite(batch).all() and torch.isfinite(single).all() and difference<=1e-5
     np.save(folder/'batch.npy',batch.numpy());np.save(folder/'singletons.npy',single.numpy())
     direct,_=wk.calculate(go,sha,name,phase,direction,depth,model,conf,grid,folder/'unused',direct=True)
     if injection is None:
      fixture=(folder,copy.deepcopy(grid),[a.clone() for a in args])
      try:wk.calculate(go,sha,name,phase,direction,depth,model,conf,grid,folder/'cache',stop_after=2)
      except wk.InjectedStop:
       injection=dict(status='INJECTED_STOP_AFTER_TWO_COMMITTED_BATCHES',receipts=len(list((folder/'cache').glob('batch-*.json'))));assert injection['receipts']==2
      else:raise AssertionError('Stop injection failed')
     cached,cc=wk.calculate(go,sha,name,phase,direction,depth,model,conf,grid,folder/'cache')
     replay,rc=wk.calculate(go,sha,name,phase,direction,depth,model,conf,grid,folder/'cache',readonly=True)
     assert direct.keys()==cached.keys()==replay.keys() and all(np.array_equal(direct[k],cached[k]) and np.array_equal(cached[k],replay[k]) for k in direct)
     assert rc['new_forwards']==0 and rc['reused_forwards']==2 and model.calls-before==9
     if len(rows)==0:assert cc['new_forwards']==0 and cc['reused_forwards']==2
     np.savez_compressed(folder/'framework-direct.npz',**direct);np.savez_compressed(folder/'framework-cached.npz',**cached)
     result=dict(status='PASS',model=name,depth=depth,phase=phase,direction=direction,cohorts=grid['batches'],max_probability_difference=difference,batch4_seconds=seconds,direct_cached_replay_exact=True,new_model_calls=model.calls-before,cache_counts=cc,replay_counts=rc)
     save(folder/'RESULT.json',result);rows.append(result);print('QUALIFIED',name,depth,phase,direction,'max_error',difference,flush=True)
     del raw,args,batch,single,direct,cached,replay
  counts.append(dict(model=name,calls=model.calls,tiles=model.tiles));del model,base,conf;gc.collect()
 # Corrupt only copied fixtures; no model is allowed to run during rejection.
 class NeverRun(torch.nn.Module):
  def forward(self,*args):raise RuntimeError('UNEXPECTED_NEURAL_FORWARD')
 folder,grid,args=fixture;original=json.loads((folder/'cache/batch-00000.json').read_text());logits=np.load(original['logits']['path']);tests=[]
 for case in ['changed_depth','changed_order','changed_model','changed_cohort','changed_input','missing_receipt','missing_logits','changed_logits']:
  target=Q/('reject-'+case);target.mkdir();rec=copy.deepcopy(original);path=target/'batch-00000.npy';np.save(path,logits);rec['logits']=pin(path);inputs=[a.clone() for a in args]
  if case=='changed_depth':rec['binding']['depth']=15
  if case=='changed_order':rec['binding']['job'][2]='reverse'
  if case=='changed_model':rec['binding']['job'][0]='hecate'
  if case=='changed_cohort':rec['cohort_yx'][0][0]+=1
  if case=='changed_input':inputs[0].flatten()[0]+=1
  if case=='changed_logits':path.write_bytes(path.read_bytes()[:-1]+b'X')
  if case!='missing_receipt':save(target/'batch-00000.json',rec)
  if case=='missing_logits':path.unlink()
  wrap=wk.CacheModel(NeverRun(),target,original['binding'],grid['batches'],readonly=True)
  try:wrap(*inputs)
  except (AssertionError,FileNotFoundError,ValueError):tests.append(dict(case=case,rejected=True))
  else:raise AssertionError('Bad cache accepted')
 assert len(rows)==32 and len(baselines)==16 and len(tests)==8
 assert sum(c['calls'] for c in counts)==288 and sum(c['tiles'] for c in counts)==768
 for r in go['pins']+go['runtime_pins']+[m['checkpoint'] for m in go['models']]:assert pin(r['path'])==r
 result=dict(status='PASS',cases=rows,worker_baseline_replays=baselines,new_forward_calls=288,tile_evaluations=768,counts_by_model=counts,probability_tolerance=1e-5,injected_stop=injection,fail_closed_tests=tests,threads=4,batch_size=4,precision='fp32 CPU',gpu_spend_GBP=0)
 save(P/'BATCH_QUALIFICATION.json',result);print('ALL_QUALIFICATION_PASS',flush=True)

if __name__=='__main__':main()
