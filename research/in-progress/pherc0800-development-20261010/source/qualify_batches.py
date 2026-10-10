"""Distinct-tile batch qualification and original-framework restart/replay tests."""
import json,time,copy,gc,shutil
import numpy as np,torch,zarr
from common import *
import worker as wk
Q=W/'qualification';Q.mkdir();go=json.loads((P/'INPUT_GO.json').read_text());go_sha=pin(P/'INPUT_GO.json')['sha256']
for r in go['pins']+go['runtime_pins']:assert pin(r['path'])==r,r['path']
wk.configure();rows=[];injection=None
for row in go['models']:
 name=row['name'];model,conf=wk.load(row);family='hecate' if name=='hecate' else 'villa';full=next(g for g in go['grids'] if g['family']==family and g['phase']=='plain');grid=copy.deepcopy(full)
 chosen={tuple(pt) for batch in full['cohorts_yx'][:2] for pt in batch};grid['patches_xywh']=[a for a in full['patches_xywh'] if (a[1],a[0]) in chosen];grid['patch_count']=len(grid['patches_xywh']);grid['cohorts_yx']=wk.cohorts(family,'plain',grid);assert len(grid['cohorts_yx'])==2
 for direction in ['forward','reverse']:
  folder=Q/(name+'-'+direction);folder.mkdir();tiles=full['patches_xywh'][:4];assert len({tuple(t) for t in tiles})==4
  if family=='villa':
   indices=np.arange(25,42);indices=indices[::-1] if direction=='reverse' else indices
   reader=wk.infer.FlatPatchReader(input_path=wk.V/'plain.zarr',resolution='0',depth_axis_first=True,height=2580,width=2580,layer_indices=indices,output_depth=17,preprocessing=conf.preprocessing)
   blocks=[wk.infer.Block(y,x,h,w) for x,y,w,h in tiles];dataset=wk.infer.FlatBlockDataset(reader=reader,blocks=blocks,patch_size=128,preprocessing=conf.preprocessing);items=[dataset[i] for i in range(4)];assert all(m[4] for a,m in items);args=[torch.stack([a for a,m in items])]
   raw=np.stack([reader.read(y,x,w,h).transpose(2,0,1) for x,y,w,h in tiles]);manual=np.stack([wk.infer.normalize_flat_patch(a,conf.preprocessing) for a in raw]);assert np.array_equal(args[0].numpy()[:,0],manual)
   probability=lambda logits:wk.infer.logits_to_probabilities(logits,image_hw=(128,128))
  else:
   v=zarr.open_group(str(wk.V/'hecate-plain.zarr'),mode='r')['0'];raw=np.stack([np.asarray(v[:,y:y+h,x:x+w]) for x,y,w,h in tiles]);raw=raw[:,::-1].copy() if direction=='reverse' else raw
   assert raw.any(axis=(1,2,3)).all();image=torch.from_numpy((raw.astype(np.float32)/model.divisor)[:,None]);support=torch.from_numpy(raw.any(1))[:,None,None].expand(-1,1,16,-1,-1);args=[image,support];probability=lambda logits:logits.float().sigmoid()
  np.save(folder/'raw.npy',raw)
  with torch.inference_mode():
   t=time.monotonic();batch=probability(model(*args));seconds=time.monotonic()-t
   singles=torch.cat([probability(model(*(a[i:i+1] for a in args))) for i in range(4)])
  error=float((batch-singles).abs().max());assert error<=1e-5 and torch.isfinite(batch).all() and torch.isfinite(singles).all()
  np.save(folder/'batch4.npy',batch.numpy());np.save(folder/'singletons.npy',singles.numpy())
  direct,_=wk.calculate(go,go_sha,name,'plain',direction,model,conf,grid,folder/'direct',direct=True)
  cache=folder/'cache'
  if name=='dnative' and direction=='forward':
   try:wk.calculate(go,go_sha,name,'plain',direction,model,conf,grid,cache,stop_after=2)
   except wk.InjectedStop:injection=dict(status='EXPECTED_STOP_AFTER_TWO_COMMITTED_BATCHES',receipts=len(list(cache.glob('batch-*.json'))),complete_map=False);assert injection['receipts']==2
   else:raise AssertionError('Fault injection failed')
  cached,counts=wk.calculate(go,go_sha,name,'plain',direction,model,conf,grid,cache)
  replay,rc=wk.calculate(go,go_sha,name,'plain',direction,model,conf,grid,cache,readonly=True)
  assert direct.keys()==cached.keys()==replay.keys() and all(np.array_equal(direct[k],cached[k]) and np.array_equal(cached[k],replay[k]) for k in direct)
  np.savez_compressed(folder/'framework-direct.npz',**direct);np.savez_compressed(folder/'framework-cached.npz',**cached)
  result=dict(status='PASS',model=name,direction=direction,tiles_xywh=tiles,batch4_seconds=seconds,max_probability_difference=error,framework_direct_cache_exact=True,cached_replay_exact=True,cache_counts=counts,replay_counts=rc)
  save(folder/'RESULT.json',result);rows.append(result);print('QUALIFIED',json.dumps(result),flush=True)
 del model,conf;gc.collect()
# Tamper real committed receipt fixtures; never mutate the original qualification.
source=Q/'dnative-forward/cache';base=source/'batch-00000.json';r0=json.loads(base.read_text());original=np.load(r0['logits']['path'])
class NeverRun(torch.nn.Module):
 def forward(self,*args):raise AssertionError('Unexpected model call during fail-closed test')
tests=[]
# Restore exact recorded input arrays for cache-only failure tests from frozen reader.
row=go['models'][0];model,conf=wk.load(row);reader=wk.infer.FlatPatchReader(input_path=wk.V/'plain.zarr',resolution='0',depth_axis_first=True,height=2580,width=2580,layer_indices=np.arange(25,42),output_depth=17,preprocessing=conf.preprocessing)
input_tensor=torch.stack([torch.from_numpy(wk.infer.normalize_flat_patch(reader.read(y,x,128,128).transpose(2,0,1),conf.preprocessing))[None] for y,x in r0['cohort_yx']]);assert wk.digest_array(input_tensor.numpy())==r0['inputs'][0]
for case in ['missing_receipt','missing_logits','corrupt_logits','wrong_binding','wrong_cohort','changed_input']:
 folder=Q/('reject-'+case);folder.mkdir();rec=copy.deepcopy(r0);path=folder/'batch-00000.npy';np.save(path,original);rec['logits']=pin(path)
 if case=='corrupt_logits':path.write_bytes(path.read_bytes()[:-1]+b'X')
 if case=='wrong_binding':rec['binding']['go_sha256']='0'*64
 if case=='wrong_cohort':rec['cohort_yx'][0][0]+=1
 if case!='missing_receipt':save(folder/'batch-00000.json',rec)
 if case=='missing_logits':path.unlink()
 tensor=input_tensor.clone()
 if case=='changed_input':tensor.flatten()[0]+=1
 wrap=wk.CacheModel(NeverRun(),folder,r0['binding'],[r0['cohort_yx']],readonly=True)
 try:wrap(tensor)
 except (AssertionError,FileNotFoundError,ValueError):tests.append(dict(case=case,rejected=True))
 else:raise AssertionError('Tampered cache accepted: '+case)
assert len(tests)==6 and injection
result=dict(status='PASS',models_and_orders=rows,distinct_tiles_per_batch=4,probability_tolerance=1e-5,original_framework_comparisons=8,exact_saved_logit_replays=8,injected_stop=injection,fail_closed_tests=tests,precision='fp32 CPU',threads=4,batch_size=4,gpu_spend_GBP=0)
save(P/'BATCH_QUALIFICATION.json',result);print('BATCH_QUALIFICATION_PASS',flush=True)
