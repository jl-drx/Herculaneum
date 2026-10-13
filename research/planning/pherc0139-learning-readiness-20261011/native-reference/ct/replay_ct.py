"""Offline native CT support replay using per-chunk Z reduction and global XY mosaics."""
import argparse,json,pathlib
import numpy as np
import intake_ct as original
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
R=pathlib.Path(__file__).resolve().parent;original.verify_freeze();plan=original.load(R/'REQUEST_PLAN.json');expected=original.load(R/'support'/'RESULTS.json');checks=[]
for site in expected['sites']:
 w=site['winding'];meta=next(x['array'] for x in plan['native_metadata'] if x['winding']==w);_,h,width=meta['shape'];mosaic=np.zeros((h,width),dtype=bool);known=np.zeros_like(mosaic)
 for row in plan['requests']:
  if row['winding']!=w:continue
  assert original.verified(row);z,cy,cx=row['chunk_zyx'];b=(R/row['relative_path']).read_bytes();chunk=np.frombuffer(b,dtype='u1').reshape(28,128,128);valid=np.any(chunk!=0,axis=0)
  y,x=cy*128,cx*128;ny,nx=min(128,h-y),min(128,width-x);assert ny>0 and nx>0
  assert not known[y:y+ny,x:x+nx].any()
  mosaic[y:y+ny,x:x+nx]=valid[:ny,:nx];known[y:y+ny,x:x+nx]=True
 for row in site['rows']:
  x,y=row['x'],row['y'];cv=mosaic[y-64:y+192,x-64:x+192];core=mosaic[y:y+128,x:x+128]
  assert known[y-64:y+192,x-64:x+192].all() and cv.shape==(256,256)
  assert original.sha(cv.astype('u1').tobytes())==row['context_valid_sha256']
  assert original.sha(core.astype('u1').tobytes())==row['core_valid_sha256']
  with np.load(R/'support'/'masks'/(row['core_id']+'.npz')) as saved:
   assert np.array_equal(saved['context_valid'],cv) and np.array_equal(saved['core_valid'],core)
   assert list(saved['core_origin_xy'])==[x,y] and list(saved['context_origin_xy'])==[x-64,y-64]
  assert int(core.sum())==row['native_valid_core_pixels'] and int(cv.sum())==row['native_valid_context_pixels']
 checks.append({'winding':w,'all_chunk_hashes_and_lengths_match':True,'all_contexts_fully_covered_by_real_chunks':True,'every_mask_matches_independent_global_mosaic':True,'verified_cores':len(site['rows'])})
original.write(a.output/'REPLAY_CT.json',{'all_passed':True,'network_access':False,'method':'Per-chunk any over full28depth first, then global XY mosaic; independent of original28×256×256 context assembly','checks':checks})
print(json.dumps(checks,indent=2))
