"""Read-only completed-run identity and numerical verification; no new inference."""
import json,itertools
import numpy as np,cv2
from common import *
r=json.loads((P/'INFERENCE_RESULTS.json').read_text());go=json.loads((P/'INFERENCE_GO.json').read_text());gsha=pin(P/'INFERENCE_GO.json')['sha256']
assert gsha=='bd10e35e367fa982a3db8bd14be581766df9e06b7629186e3dba54f318cfc05e'
for rec in go['pins']+go['runtime_pins']+[row['checkpoint'] for row in go['models']]:assert pin(rec['path'])==rec,rec['path']
assert r['status']=='PASS' and len(r['maps'])==16 and r['all_saved_logit_replays_exact']
expected=set(itertools.product([m['name'] for m in go['models']],['plain','padded'],['forward','reverse']));assert {(m['model'],m['phase'],m['direction']) for m in r['maps']}==expected
support=np.load(W/'output-support.npz');assert support['common'].sum()==1875648
batch_count=tile_count=0
for row in r['maps']:
 fam='hecate' if row['model']=='hecate' else 'villa';grid=next(g for g in go['grids'] if (g['family'],g['phase'])==(fam,row['phase']));folder=W/'inference'/('-'.join([row['model'],row['phase'],row['direction']]))
 assert json.loads((folder/'COMPLETE.json').read_text())==row and row['status']=='PASS' and row['go_sha256']==gsha
 assert row['read_only_replay']['status']=='PASS' and row['read_only_replay']['new_forwards']==0 and row['read_only_replay']['reused_forwards']==len(grid['cohorts_yx'])
 assert row['counts']['model_calls']==len(grid['cohorts_yx']) and row['counts']['model_tiles']==sum(map(len,grid['cohorts_yx']))
 for rec in row['outputs'].values():assert pin(rec['path'])==rec
 arrays={k:np.load(rec['path'],mmap_mode='r') for k,rec in row['outputs'].items()};aligned=arrays['aligned'];mask=support[fam+'_'+row['phase']]
 assert aligned.shape==(1536,1536) and np.isfinite(aligned).all() and aligned.min()>=0 and aligned.max()<=1
 assert np.isnan(arrays['supported'][~mask]).all() and np.array_equal(arrays['supported'][mask],aligned[mask])
 assert np.array_equal(arrays['display'],np.rint(aligned*255).astype(np.uint8))
 if fam=='villa':
  num=arrays['numerator'].copy();den=arrays['denominator'];np.divide(num,den,out=num,where=den>1e-6);np.clip(num,0,1,out=num);assert np.array_equal(num,arrays['unquantized']);assert np.array_equal((num*255).astype(np.uint8),arrays['raw_grid']);assert np.array_equal(arrays['raw_grid'].astype(np.float32)/255,aligned)
 else:
  py,px=(0,0) if row['phase']=='plain' else (40,200);native=cv2.resize(arrays['raw_grid'].astype(np.float32)/255,(2580+px,2580+py),interpolation=cv2.INTER_LINEAR);assert np.array_equal(native[512+py:2048+py,512+px:2048+px],aligned)
 receipts=sorted(folder.glob('batch-*.json'));assert len(receipts)==len(grid['cohorts_yx'])
 for ix,f in enumerate(receipts):
  b=json.loads(f.read_text());assert b['index']==ix and b['binding']==dict(go_sha256=gsha,job=[row['model'],row['phase'],row['direction']]);assert b['cohort_yx']==grid['cohorts_yx'][ix];assert pin(b['logits']['path'])==b['logits'];a=np.load(b['logits']['path']);assert np.isfinite(a).all() and a.shape[0]==len(b['cohort_yx']);assert all(v['shape'][0]==a.shape[0] for v in b['inputs'])
 batch_count+=len(receipts);tile_count+=sum(map(len,grid['cohorts_yx']))
assert batch_count==r['total_unique_model_batches']==3454 and tile_count==r['total_model_tiles']==13250
samples=np.load(W/'render-verification-samples.npz');d=np.abs(samples['actual'].astype(np.int16)-samples['independent'].astype(np.int16));stats=dict(samples=int(d.size),exact_fraction=float((d==0).mean()),mean_abs=float(d.mean()),max_abs=int(d.max()));rv=json.loads((P/'RENDER_VERIFICATION.json').read_text());assert stats==rv['pooled'] and all(x['pass'] for x in rv['sampling_checks'])
result=dict(status='PASS',frozen_input_runtime_and_model_pins=len(go['pins'])+len(go['runtime_pins'])+len(go['models']),completed_maps=16,committed_batch_receipts=batch_count,model_tiles=tile_count,all_saved_conversions_exact=True,all_unknown_masks_preserved=True,recorded_exact_framework_replays=16,new_model_forwards=0,render_sample_metrics_recomputed=stats,common_support_pixels=int(support['common'].sum()),gpu_spend_GBP=0)
save(P/'COMPLETION_VERIFICATION.json',result);print(json.dumps(result))
