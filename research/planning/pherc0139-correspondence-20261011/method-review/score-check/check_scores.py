#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,importlib.util,itertools
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('scored',ROOT/'structural_scores_reviewed.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def pearson(a,b):
 a=np.array(a,dtype=np.float64).ravel();b=np.array(b,dtype=np.float64).ravel()
 a-=sum(a)/len(a);b-=sum(b)/len(b)
 return float(np.sum(a*b)/(np.sqrt(np.sum(a*a))*np.sqrt(np.sum(b*b))))

def main():
 rng=np.random.default_rng(8120139)
 # Unsmoothed independent random field, different seed and offset/scale fromauthorfixture.
 field=rng.normal(size=(40,290,290));fine=field[10:30,9:281,9:281]
 cases=[];maximum_error=0
 B=np.array([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]])
 for shift in ((1,-2,3),(-3,4,-1),(0,0,6)):
  dz,dy,dx=shift;native=5.1*field[10+dz:30+dz,9+dy:281+dy,9+dx:281+dx]-23
  scores=m.landscapes(native,fine)
  # Explicit index arithmetic independent of landscapes' slice implementation.
  for si,(cx,cy) in enumerate(m.FIT_XY+m.CHECK_XY):
   n=native[np.ix_(np.arange(6,14),np.arange(cy-8,cy+24),np.arange(cx-8,cx+24))]
   indices=itertools.product(range(13),repeat=3) if si==0 and shift==(1,-2,3) else ((0,0,0),(6,6,6),(12,12,12),(7,3,9),(6+dz,6+dy,6+dx))
   for iz,iy,ix in indices:
    q=fine[np.ix_(np.arange(iz,iz+8),np.arange(cy-14+iy,cy+18+iy),np.arange(cx-14+ix,cx+18+ix))]
    v=pearson(n,q);err=abs(v-scores[si,iz,iy,ix]);maximum_error=max(maximum_error,err)
    assert err<1e-11,(si,shift,(iz,iy,ix),err)
  result=m.summarise(scores,B,[B]*8)
  assert tuple(result['fit_translation_zyx'])==shift
  assert all(tuple(x['independent_best_zyx'])==shift for x in result['supports'])
  assert result['structural_screen_pass']==(shift!=(0,0,6))
  cases.append({'injected_shift_zyx':shift,'fit_shift_zyx':result['fit_translation_zyx'],'screen_pass':result['structural_screen_pass'],'criteria':result['criteria']})
 # Periodic profile must fail unique-margin screen even when its absolute NCC is1.
 repeated=np.tile(rng.normal(size=(4,272,272)),(5,1,1));ps=m.landscapes(repeated,repeated);pr=m.summarise(ps,B,[B]*8)
 assert not pr['structural_screen_pass'] and not pr['criteria']['all_check_margin_at_least_005']
 # Constant search givesNaNs which peak correctly refuses; constanttemplate rejected.
 constant=[]
 for a,b,kind in ((np.ones((20,44,44)),np.ones((8,32,32)),'constant_template'),(np.ones((20,44,44)),rng.normal(size=(8,32,32)),'constant_search')):
  failed=False
  try:m.peak(m.ncc_landscape(a,b))
  except ValueError:failed=True
  assert failed;constant.append(kind)
 # All mapped fine interpolation+binned-Zfilter footprints are disjoint byboundingboxes.
 support=[]
 for site in ('site1','site2','site3'):
  q=np.load(BASE/'geometry-review/pullback'/site/'queries_uvk.npy',mmap_mode='r')
  boxes=[]
  for cx,cy in m.FIT_XY+m.CHECK_XY:
   t=np.asarray(q[:,cy-14:cy+30,cx-14:cx+30])
   lo=np.floor(t.min(axis=(0,1,2))).astype(int);hi=np.floor(t.max(axis=(0,1,2))).astype(int)+1
   lo[2]-=5;hi[2]+=5 # trilinear sample of radius5-smoothed fineZvolume
   boxes.append([lo.tolist(),hi.tolist()])
  collisions=[]
  for i,j in itertools.combinations(range(8),2):
   alo,ahi=np.array(boxes[i]);blo,bhi=np.array(boxes[j])
   if np.all(np.maximum(alo,blo)<=np.minimum(ahi,bhi)):collisions.append([i,j])
  assert not collisions
  support.append({'site':site,'source_boxes_inclusive_uvk_after_Zfilter':boxes,'overlapping_boxes':collisions})
 # Quantify effectofcorrect support-local basis nearstrict2voxel neighbourhood.
 local=json.loads((BASE/'geometry-review/pullback/LOCAL_BASES.json').read_text());metrics=[]
 for site in local['sites']:
  C=np.asarray(site['core_basis_columns_depth_y_x_nativeXYZ']);dd=m.SHIFTS@C.T;corefar=np.linalg.norm(dd,axis=-1)>2
  rows=[]
  for i,s in enumerate(site['supports']):
   L=np.asarray(s['basis_columns_depth_y_x_nativeXYZ']);d=m.SHIFTS@L.T;dist=np.linalg.norm(d,axis=-1);far=dist>2
   near=(np.abs(dist-2)<1e-6)
   rows.append({'support':i,'far_mask_entries_changed_vs_core_at_zero_shift':int(np.sum(far!=corefar)),'entries_within_1e_6_of_threshold2':int(near.sum()),'depth_basis_norm':float(np.linalg.norm(L[:,0]))})
  metrics.append({'site':site['site'],'supports':rows})
 result={'status':'PASS','reviewed_code_sha256':hashlib.sha256((ROOT/'structural_scores_reviewed.py').read_bytes()).hexdigest(),'independent_direct_pearson_max_abs_error':maximum_error,'known_shift_cases':cases,'periodic_case_criteria':pr['criteria'],'constant_cases_rejected':constant,'mapped_support_nonoverlap':support,'local_basis_threshold_diagnostics':metrics,'limitations':['NoCTorlabels;synthetic arithmetic andgeometry support only.','Oneinteger shiftinsurface coordinates isnot oneglobalXYZtranslation.','Support-centre Jacobians arelocal linear metrics; reportthreshold-near decisions without claimofcalibratedphysicalconfidence.']}
 (ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','independent_direct_pearson_max_abs_error','known_shift_cases')},indent=2))
if __name__=='__main__':main()
