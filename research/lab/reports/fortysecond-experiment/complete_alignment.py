"""Read completed score cubes and assess remaining pairs without relaxing any gate."""
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from common import *
from alignment import ncc_grid,peak,masks,sample,basis
def main():
 started=time.monotonic();go=json.loads((HERE/'ALIGNMENT_CONTINUATION_GO.json').read_text())
 for item in go['pins']:verify(item)
 out=HERE/'alignment';selection=json.loads((HERE/'geometry/SELECTION.json').read_text());receipt=json.loads((HERE/'ct/COMPLETE.json').read_text());meta=json.loads((PRIOR/'PAIR_FEASIBILITY.json').read_text());A=np.asarray(meta['forward']['matrix']);geom=np.load(HERE/'geometry/old-xyz.npy');parts=masks();fitmask=np.logical_or.reduce([r['mask'] for r in parts if r['role']=='fit']);rows=[];new_count=0
 lattice=lambda n,origin:np.moveaxis(np.indices((n,n,n),dtype=float),0,-1)[...,::-1]+np.asarray(origin)-n//2
 for c in selection['candidates']:
  block={r['volume']:r for r in receipt['arrays'] if r['pair']==c['id']};points=lattice(32,c['old_xyz']);mapped=points@A[:,:3].T+A[:,3];local=mapped[...,::-1]-np.asarray(block['new']['start_zyx']);coverage=bool(((local>=6)&(local<=249)).all());axes=basis(geom,c['old_grid_yx']);normal=axes[2]
  if not coverage:
   rows.append(dict(id=c['id'],nominal_role=c['nominal_role'],status='NOT_ADMITTED_COVERAGE',sample_bounds_zyx=[local.min((0,1,2)).tolist(),local.max((0,1,2)).tolist()],required_bounds=[6,249],numerical_screen_passes=False,physical_registration_admitted=False,geometry_basis_old_xyz=axes.tolist(),fit=None));continue
  prior=c['id'] in ['pair01','pair02'];paths=[out/(c['id']+'-fit-ncc.npy')]+[out/(c['id']+f'-check{j}-ncc.npy') for j in range(4)]
  if prior:
   scores=[np.load(path) for path in paths]
  else:
   need(not any(p.exists() for p in paths),'No repeated score search');old=np.load(verify(block['old']['array']));new=np.load(verify(block['new']['array']));blur=gaussian_filter(new.astype(np.float32),(7.91/2.4)/2,truncate=3)
   fine,valid=sample(blur,mapped,block['new']['start_zyx']);real,v=sample(old,lattice(48,c['old_xyz']),block['old']['start_zyx']);need(valid.all() and v.all(),'Full sampling coverage');fine-=gaussian_filter(fine,1,truncate=2);real-=gaussian_filter(real,1,truncate=2)
   scores=[ncc_grid(fine,real,fitmask)]+[ncc_grid(fine,real,r['mask']) for r in parts if r['role']=='check'];new_count+=1
   for p,a in zip(paths,scores):np.save(p,a,allow_pickle=False)
  fitted=peak(scores[0]);checks=[]
  for r,part in zip(map(peak,scores[1:]),[p for p in parts if p['role']=='check']):
   delta=(np.asarray(r['refined_shift_zyx'])-fitted['refined_shift_zyx'])[::-1];n=float(np.dot(delta,normal));t=float(np.linalg.norm(delta-n*normal));r.update(corner_zyx=part['corner_zyx'],residual_xyz=delta.tolist(),normal_residual_coarse_voxels=n,tangential_residual_coarse_voxels=t);checks.append(r)
  ok=not fitted['boundary'] and all(not r['boundary'] and r['NCC']>=.5 and r['margin']>=.05 for r in checks) and np.median([r['tangential_residual_coarse_voxels'] for r in checks])<=.5 and max(r['tangential_residual_coarse_voxels'] for r in checks)<=1
  rows.append(dict(id=c['id'],nominal_role=c['nominal_role'],status='COMPLETED_RETAINED' if prior else 'COMPLETED_ONCE',public_mapping_fit_NCC=float(scores[0][8,8,8]),fit=fitted,checks=checks,numerical_screen_passes=bool(ok),geometry_basis_old_xyz=axes.tolist(),independent_landmarks_verified=False,wrap_identity_verified=False,physical_registration_admitted=False,score_files=[pin(p) for p in paths]))
 result=dict(status='COMPLETED_WITH_COVERAGE_REJECTIONS',created_at_utc=now(),rows=rows,all_six_numerical_screens_pass=all(r['numerical_screen_passes'] for r in rows),physical_registration_established=False,source_reference_labels_established=False,new_model_calls=0,completed_before_guard=2,correlation_regions_repeated=0,remaining_regions_computed_once=new_count,elapsed_seconds=time.monotonic()-started)
 save(out/'RESULTS.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
