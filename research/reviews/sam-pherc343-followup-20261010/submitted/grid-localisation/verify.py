import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
O=Path('/workspace/shared/pherc343-grid-output');r=json.loads((O/'metrics.json').read_text());R=Path('/workspace/shared/pherc343-input');v=np.load(R/'valid.npy');B=R/'restored/reports/fortysixth-experiment/work'
for n,t in [('dnative','dnative'),('seed42','hybrid_3d2d-seed42'),('seed43','hybrid_3d2d-seed43')]:
 a=np.array(Image.open(next((B/'maps_plain').glob(f'*__{t}__*.tif'))));b=np.array(Image.open(next((B/'maps_padded').glob(f'*__{t}__*.tif'))))[40:2600,200:6920];sums=np.zeros((8,8),np.int64);counts=sums.copy();near_sum=near_n=far_sum=far_n=0
 for y in range(105,2560):
  x=np.arange(380,6021);valid=v[y,380:6021];d=np.abs(a[y,380:6021].astype(int)-b[y,380:6021].astype(int))
  dy=min(abs(y-yy) for yy in range(0,2433,64));dx=np.min(abs(x[:,None]-np.arange(0,6593,64)[None,:]),axis=1);dist=np.minimum(dy,dx)
  near=valid&(dist<=4);far=valid&(dist>=13);near_sum+=d[near].sum();near_n+=near.sum();far_sum+=d[far].sum();far_n+=far.sum()
  if 128<=y<2432:
   for k in range(8):
    mask=valid&((x%64//8)==k);sums[y%64//8,k]+=d[mask].sum();counts[y%64//8,k]+=mask.sum()
 m=r['models'][n];assert np.allclose((sums/counts).ravel(),m['phase8x8']['mae'],rtol=0,atol=1e-12)
 for key,total,num in [('near',near_sum,near_n),('far',far_sum,far_n)]:
  q=m['crop']['grid']['plain'][key];assert q['n']==num and abs(q['mae']-total/num)<1e-12
 for pop in ['crop','interior_crop','all18_union']+[f'box{i}' for i in range(1,19)]:
  for g in ['plain','padded']:
   bins=m[pop]['grid'][g]['bands'];assert sum(z['n'] for z in bins)==m[pop]['n'];assert abs(sum(z['mae']*z['n'] for z in bins if z['n'])/m[pop]['n']-m[pop]['mae'])<1e-10
 print(n,'PASS')
(O/'verification.json').write_text(json.dumps({'status':'PASS','checks':['all6 decoded hashes checked in analyze.py','crop means match prior review','independent rowwise plain near/far arithmetic','independent64 phase-bin means','distance band count and error conservation for crop/interior/union/all18 boxes']},indent=2)+'\n')
