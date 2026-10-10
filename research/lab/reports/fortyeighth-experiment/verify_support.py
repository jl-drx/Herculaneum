"""Independent archived-point, geometric bound and saved-mask checks; no inference."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import tifffile
from scipy.ndimage import label
from scipy.spatial import cKDTree
from support import reconstruct,corner_envelope,context_core,largest_square
P=Path(__file__).resolve().parent;LAB=P.parents[1];g=json.loads((P/'GO.json').read_text());rows=json.loads((P/'support-results/RESULTS.json').read_text())
files={r['segment']:r['files'] for r in g['meshes']}
checks=[]
# Prior point receipts were independently native-CT checked in earlier studies.
for report in ['reports/first-experiment/native-ct-consistency.json','reports/third-experiment/native-ct-consistency.json','reports/fifth-experiment/native-check/frozen-plan.json']:
 d=json.loads((LAB/report).read_text());f=files[d['segment_id']];meta=json.loads(Path(f['mesh.meta.json']).read_text());shape=json.loads(Path(f['surface.level0.zarray.json']).read_text())['shape'][1:]
 mesh=np.stack([tifffile.imread(f[f'mesh.{a}.tif']) for a in 'xyz'],-1);base,normal,_,valid=reconstruct(mesh,meta['scale'],shape)
 for row in d['points']:
  x,y=row.get('surface_xy',row.get('source_xy'));assert valid[y,x]
  eb=float(np.abs(base[y,x]-row['base_ct_xyz']).max());en=float(np.abs(normal[y,x]-row['normal_xyz_flipped']).max());assert eb<=.002 and en<=1e-6
  checks.append({'segment':d['segment_id'],'xy':[x,y],'base_error':eb,'normal_error':en})
 del base,normal
# Independent planar analytic fixture and invalid normal stencil.
y,x=np.indices((8,9));mesh=np.stack([1000+x*20,2000+y*20,3000+x*4+y*7],-1).astype(np.float32)
base,n,bi,valid=reconstruct(mesh,[.05,.05],(160,180));normal=np.cross([20,0,4],[0,20,7]);normal=-normal/np.linalg.norm(normal)
assert np.max(np.abs(n[valid]-normal))<1e-6
mesh[3,3]=-1;_,_,bi2,v2=reconstruct(mesh,[.05,.05],(160,180));assert (bi2&~v2).any() and not (v2&~bi2).any()
# The spherical exclusion is an analytical bound including both interpolation corners.
assert g['surface_bound_voxels']>32.5+15+2*np.sqrt(3)+.02
# Check every discrete layer/corner at fixed synthetic positions is in endpoint AABB.
b=np.array([[100.1,200.2,300.3],[100.99,200.99,300.99]],np.float32);nn=np.array([[.6,0,.8],[-1,0,0]],np.float32)
lo,hi=corner_envelope(b,nn)
for t in np.arange(66)-32.5:
 q=b.astype(float)+t*nn
 for corner in itertools.product([0,1],repeat=3):
  indices=np.floor(q)+corner;assert (indices>=lo).all() and (indices<=hi).all()
# Integral rectangles provide an independent context erosion implementation.
def integral_core(a,r):
 z=np.pad((~a).astype(np.int64),((r,r),(r,r)),constant_values=1)
 s=np.pad(z,((1,0),(1,0))).cumsum(0).cumsum(1);n=2*r+1
 return (s[n:,n:]-s[:-n,n:]-s[n:,:-n]+s[:-n,:-n])==0
rng=np.random.default_rng(20261010);fixtures=[np.ones((35,41),bool),rng.random((35,41))>.03,np.zeros((35,41),bool)]
for a in fixtures:
 for r in [0,1,4]:assert np.array_equal(context_core(a,r),integral_core(a,r))
 for n in range(1,min(a.shape)+1):
  windows=np.lib.stride_tricks.sliding_window_view(a,(n,n));found=windows.all((-1,-2)).any()
  assert found==(n<=largest_square(a)['side'])
results=[]
for row in rows['meshes']:
 z=np.load(P/'support-results'/f"{row['segment']}.npz");v=z['inside_CT'];dist=z['nearest_old_distance'];near=z['old_surface_near'];bits=z['old_CT_box_bits']
 assert np.array_equal(near,z['bilinear_valid']&(dist<=g['surface_bound_voxels']))
 support=v&~near&(bits==0);assert np.array_equal(support,z['point_support'])
 expected=integral_core(support,g['context_radius_native_pixels']);assert np.array_equal(expected,z['context_core'])
 assert int(support.sum())==row['point_support'] and int(expected.sum())==row['context_core']
 # Full-square criterion checked independently with integral sums for size128.
 good=np.pad(expected.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1);k=g['minimum_core_square']
 cells=good[k:,k:]-good[:-k,k:]-good[k:,:-k]+good[:-k,:-k]
 any128=bool((cells==k*k).any());assert any128==(row['unrestricted_best_square']['side']>=128)
 for component in [c for c in rows['components_in_selection_order'] if c['segment']==row['segment']]:
  m=expected&(z['candidate_component']==component['component']);assert int(m.sum())==component['usable_core_pixels']
  box=component['box_xywh']
  if box:
   x,y,w,h=box;assert m[y:y+h,x:x+w].all()
 results.append({'segment':row['segment'],'all_masks_counts_and_context_exact':True,'any_unrestricted_128_square':any128})
out={'status':'PASS','archived_point_checks':checks,'synthetic_checks':['analytic normal','missing-stencil exclusion','slab/corner bound','layer endpoint AABB','nine integral context fixtures','exhaustive square-size fixtures'],
 'saved_masks':results,'scope':'Numerical support and conservative-admission verification; no new raw-CT values or physical-sheet truth.'}
with (P/'VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','archived_points':len(checks),'mesh_masks_verified':len(results)}))
