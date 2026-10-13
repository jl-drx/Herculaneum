import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from audit import largest,exact_context
rng=np.random.default_rng(8066)
checks=[]
# Exhaustive square oracle independent of the integral-image binary search.
for h,w in [(3,5),(8,7),(11,9)]:
 for _ in range(30):
  a=rng.random((h,w))>.25;expected=(0,None)
  for n in range(1,min(h,w)+1):
   candidates=[(y,x) for y in range(h-n+1) for x in range(w-n+1) if a[y:y+n,x:x+n].all()]
   if candidates:
    y,x=min(candidates);expected=(n,[x,y,n,n])
  actual=largest(a);assert (actual['side'],actual['xywh'])==expected
checks.append('90 randomized largest-square/tie-break cases match exhaustive oracle')
# Independent per-pixel context unions include irregular final starts.
a=np.ones((270,277),bool);a[139,140]=False;a[3,3]=False
actual,patches=exact_context(a)
for y,x in rng.integers([0,0],a.shape,size=(500,2)):
 ys=list(range(0,270-127,64));xs=list(range(0,277-127,64));ys.append(270-128);xs.append(277-128)
 expected=all(a[py:py+128,px:px+128].all() for py in set(ys) for px in set(xs) if py<=y<py+128 and px<=x<px+128)
 assert actual[y,x]==expected
checks.append('500 fixed-grid pixel tests match direct all-covering-patches oracle, including nonstride final starts')
# Endpoint enclosure checked against all 66 float32 layer samples.
b=rng.uniform([3000,5000,11000],[5000,7500,13500],(10000,3)).astype(np.float32)
n=rng.normal(size=(10000,3)).astype(np.float32);n/=np.linalg.norm(n,axis=1)[:,None]
lo=np.floor(b.astype(float)-32.5*np.abs(n.astype(float))-.01).astype(int)
hi=np.floor(b.astype(float)+32.5*np.abs(n.astype(float))+.01).astype(int)+1
for k in range(66):
 sample=b+np.float32(k-32.5)*n;floor=np.floor(sample).astype(int)
 assert (floor>=lo).all() and (floor+1<=hi).all()
checks.append('10,000 randomized normals x 66 layers: float32 trilinear corners enclosed by endpoint integer AABB')
Path(__file__).with_name('unit-checks.json').write_text(json.dumps({'passed':checks},indent=2));print('\n'.join(checks))
