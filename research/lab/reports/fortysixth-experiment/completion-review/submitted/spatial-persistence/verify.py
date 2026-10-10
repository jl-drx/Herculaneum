import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,itertools
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
O=Path('/workspace/shared/pherc343-persistence-output');R=Path('/workspace/shared/pherc343-input');B=R/'restored/reports'
z=json.loads((O/'metrics.json').read_text());v=np.load(R/'valid.npy');src=np.load(next((B/'fortyfifth-experiment/ph343-mesh-replay').glob('*.npz')))['source'];own=np.isin(src.repeat(20,0).repeat(20,1)[:2560,:6720],[1,4]);maps={}
for n,t in [('dnative','dnative'),('seed42','hybrid_3d2d-seed42'),('seed43','hybrid_3d2d-seed43')]:
 for p in ['plain','padded']:
  a=np.array(Image.open(next((B/f'fortysixth-experiment/work/maps_{p}').glob(f'*__{t}__*.tif'))));maps[n+'_'+p]=a if p=='plain' else a[40:2600,200:6920]
 maps[n+'_merged']=np.where(own,maps[n+'_padded'],maps[n+'_plain'])
pairs=[(n+' phase',n+'_plain',n+'_padded') for n in ['dnative','seed42','seed43']]+[(a+' / '+b,a+'_merged',b+'_merged') for a,b in itertools.combinations(['dnative','seed42','seed43'],2)]
checks=0
for r in z['results']:
 x,y,X,Y=r['bounds_xyxy'];sl=np.s_[y:Y,x:X];valid=v[sl]
 for f in r['fractions']:
  points={};masks={}
  for n,m in f['selections'].items():
   a=maps[n][sl];b=(a>=m['threshold_uint8'])&valid;assert b.sum()==m['selected'];assert np.count_nonzero(valid&(a>m['threshold_uint8']))==m['strict_above_count'];assert m['strict_above_count'] < np.ceil(f['q']*valid.sum()) <= b.sum();masks[n]=b
   if r['id']!='published_crop':
    points[n]=np.argwhere(b);tree=cKDTree(points[n]);d=tree.query(points[n],workers=1)[0];assert np.all(d==0)
  for p,a,b in pairs:
   m=f['comparisons'][p];A=masks[a];B_=masks[b];inter=(A.astype(np.uint8)*B_.astype(np.uint8)).sum();assert np.isclose(2*inter/(A.sum()+B_.sum()),m['dice']);assert np.isclose(inter/(A.sum()+B_.sum()-inter),m['jaccard']);checks+=1
   if r['id']!='published_crop':
    pa,pb=points[a],points[b];d=np.r_[cKDTree(pb).query(pa,workers=1)[0],cKDTree(pa).query(pb,workers=1)[0]];assert all(np.isclose((d<=rad).mean(),m['within_radius_native_px'][str(rad)]['pooled_symmetric']) for rad in [0,2,4,8]);assert np.isclose(d.mean(),m['symmetric_nearest_mean_px']);assert np.isclose(np.percentile(d,95),m['symmetric_nearest_p95_px']);assert np.isclose(np.linalg.norm(pa.mean(0)-pb.mean(0)),m['centroid_displacement_px'])
 print('verified',r['id'],flush=True)
(O/'verification.json').write_text(json.dumps({'status':'PASS','comparison_checks':checks,'all_supports_overlap':'Independent arithmetic from reloaded maps and threshold selections matches.','all_18_boxes_positions':'Independent cKDTree nearest distances (both directions), p95 and centroids match all 324 box/pair/rank comparisons; exact self distances zero.','whole_crop_positions':'Computed by EDT in primary script, not independently recomputed; overlap recomputed at all fractions.','limitations':'Implementation verification only; no new biological evidence.'},indent=2)+'\n')
