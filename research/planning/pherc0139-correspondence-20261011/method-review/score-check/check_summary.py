#!/usr/bin/env python3
"""Independent scalar check of support-specific physical-score bookkeeping."""
from pathlib import Path
import importlib.util,json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('score',ROOT/'structural_scores_reviewed.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
local=json.loads((BASE/'geometry-review/pullback/LOCAL_BASES.json').read_text())
rng=np.random.default_rng(9139040);states=[];maxerror=0
for site in local['sites']:
 bases=[np.asarray(e['basis_columns_depth_y_x_nativeXYZ']) for e in site['supports']]
 core=np.asarray(site['core_basis_columns_depth_y_x_nativeXYZ'])
 scores=rng.uniform(-.2,.2,(8,13,13,13));scores[:4,8,5,7]=.9
 scores[4,8,5,7]=.8;scores[5,7,5,7]=.85;scores[6,8,6,7]=.8;scores[7,8,5,10]=.9
 r=m.summarise(scores,core,bases);fit=tuple(r['fit_translation_zyx']);assert fit==(2,-1,1)
 for i,(B,entry) in enumerate(zip(bases,r['supports'])):
  n=B[:,0]/np.linalg.norm(B[:,0]);normal=[];distinct=[]
  for iz,iy,ix in itertools.product(range(13),repeat=3):
   shift=np.asarray((iz-6,iy-6,ix-6));delta=B@(shift-np.asarray(fit))
   distance=float(np.sqrt(sum(v*v for v in delta)));projection=float(sum(a*b for a,b in zip(delta,n)))
   if distance>2:distinct.append((float(scores[i,iz,iy,ix]),(iz-6,iy-6,ix-6)))
   if abs(projection)>2:normal.append((float(scores[i,iz,iy,ix]),(iz-6,iy-6,ix-6)))
  bestfar=max(distinct,key=lambda t:t[0]);bestnormal=max(normal,key=lambda t:t[0])
  assert tuple(entry['distinct_competitor_zyx'])==bestfar[1]
  assert tuple(entry['normal_competitor_zyx'])==bestnormal[1]
  assert entry['distinct_competitor_score']==bestfar[0]
  assert entry['normal_competitor_score']==bestnormal[0]
  delta=B@(np.asarray(entry['independent_best_zyx'])-fit);projection=float(sum(a*b for a,b in zip(delta,n)))
  tangent=np.sqrt(sum(v*v for v in delta-projection*n));margin=float(scores[(i,)+(tuple(s+6 for s in fit))]-bestfar[0])
  for key,v in (('check_fit_normal_delta_native_vox',projection),('check_fit_tangential_delta_native_vox',tangent),('margin',margin)):
   error=abs(entry[key]-v);maxerror=max(maxerror,error);assert error<1e-11
 states.append({'site':site['site'],'support_metrics_and_competitor_indices_agree':True,'criteria':r['criteria']})
result={'status':'PASS','sites':states,'maximum_scalar_metric_difference':maxerror,'limitation':'Exact bookkeeping comparison for declared support-centre metric; not physicalregistration or nonlinear finite-displacementvalidation.'}
(ROOT/'SUMMARY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
