"""Evaluate frozen geometry/context bounds with no scan or detector access."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import tifffile
from scipy.spatial import cKDTree
from scipy.ndimage import label
from support import reconstruct,corner_envelope,context_core,largest_square
P=Path(__file__).resolve().parent;LAB=P.parents[1]
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
g=json.loads((P/'GO.json').read_text())
for r in g['pins']+g['current_sources']:assert sha(r['path'])==r['sha256'],r['path']
O=P/'support-results';O.mkdir()
meshes={r['segment']:r['files'] for r in g['meshes']}
centres=[]
for seg,(x,y,w,h) in g['previous_surface_extents']:
 f=meshes[seg];a=np.load(f['full-surface-ct-coordinates.xyz.npy'],mmap_mode='r');v=np.load(f['full-surface-valid-mask.npy'],mmap_mode='r')
 centres.append(a[y:y+h,x:x+w][v[y:y+h,x:x+w]].astype(float))
centres=np.concatenate(centres);tree=cKDTree(centres);assert len(centres)==3260832
rows=[];components=[]
for seg in g['candidate_segments']:
 start=time.monotonic();f=meshes[seg];meta=json.loads(Path(f['mesh.meta.json']).read_text());array=json.loads(Path(f['surface.level0.zarray.json']).read_text());shape=tuple(array['shape'][1:])
 export=json.loads(Path(f['source-render-export-metadata.json']).read_text())['creation_info']['provenance']['parameters']
 for name,value in {'commit':'530584554c031c982ba3aa243e1ea39c4bef9818','scale':'1','num-slices':'31','flip-normals':'true','rotate':'0','flip':'-1','crop-x':'0','crop-y':'0','crop-width':'0','crop-height':'0'}.items():assert export[name]==value
 for name in ['affine-transform','affines','affine-matrix','scale-segmentation']:assert export[name]==''
 mesh=np.stack([tifffile.imread(f[f'mesh.{a}.tif']) for a in 'xyz'],-1)
 base,normal,bilinear,normal_valid=reconstruct(mesh,meta['scale'],shape)
 oldbase=np.load(f['full-surface-ct-coordinates.xyz.npy'],mmap_mode='r');oldvalid=np.load(f['full-surface-valid-mask.npy'],mmap_mode='r')
 assert np.array_equal(bilinear,oldvalid)
 maxerror=float(np.abs(base[bilinear]-oldbase[bilinear]).max());assert maxerror<=.01,(seg,maxerror)
 lo,hi=corner_envelope(base,normal)
 inside=normal_valid&(lo>=0).all(-1)&(hi<np.array(g['target_shape_zyx'][::-1])).all(-1)
 near=np.zeros(shape,bool);dist=np.full(shape,np.nan,np.float64)
 # All valid candidate bases queried at full native resolution, no20px approximation.
 dist[bilinear]=tree.query(base[bilinear].astype(float),workers=4)[0]
 near[bilinear]=dist[bilinear]<=g['surface_bound_voxels']
 touched=np.zeros(shape,np.uint8)
 for i,box in enumerate(g['previous_CT_boxes']):
  overlap=normal_valid&(hi>=box['lo_xyz']).all(-1)&(lo<=box['hi_xyz_inclusive']).all(-1)
  touched[overlap]|=np.uint8(1<<i)
 separated=inside&~near&(touched==0)
 core=context_core(separated,g['context_radius_native_pixels'])
 prior=np.load(LAB/'reports/fortyseventh-experiment/target-preparation-v2'/f'{seg}.npz')
 labs=prior['far_component_labels'];counts=np.bincount(labs.ravel());counts[0]=0
 # Nearest sampled point's20px cell; never silently treat its support as measured.
 iy=np.clip((np.arange(shape[0])+10)//20,0,labs.shape[0]-1);ix=np.clip((np.arange(shape[1])+10)//20,0,labs.shape[1]-1)
 ids=labs[iy[:,None],ix[None,:]]
 for component in np.flatnonzero(counts):
  mask=core&(ids==component);best=largest_square(mask)
  components.append({'segment':seg,'component':int(component),'phase47_sample_count':int(counts[component]),'usable_core_pixels':int(mask.sum()),**best,'qualifies_minimum_core':best['side']>=g['minimum_core_square']})
 np.savez_compressed(O/(seg+'.npz'),bilinear_valid=bilinear,normal_valid=normal_valid,inside_CT=inside,nearest_old_distance=dist,
                    old_surface_near=near,old_CT_box_bits=touched,point_support=separated,context_core=core,candidate_component=ids)
 row={'segment':seg,'shape_yx':shape,'max_base_error_voxels':maxerror,'bilinear_valid':int(bilinear.sum()),'normal_valid':int(normal_valid.sum()),'inside_CT':int(inside.sum()),
      'outside_old_surface_bound':int((inside&~near).sum()),'old_CT_intersections_on_geometry_valid':int((inside&(touched>0)).sum()),
      'point_support':int(separated.sum()),'context_core':int(core.sum()),'unrestricted_best_square':largest_square(core),'seconds':time.monotonic()-start}
 rows.append(row);print(json.dumps(row),flush=True)
lead=max((r for r in components if r['segment']=='20251028225813'),key=lambda r:r['phase47_sample_count'])
ordered=[lead]+sorted([r for r in components if r is not lead],key=lambda r:(-r['phase47_sample_count'],r['segment'],r['component']))
selected=next((r for r in ordered if r['qualifies_minimum_core']),None)
result={'status':'PASS','decision':'GEOMETRY_SUPPORT_CORE_QUALIFIED' if selected else 'NO_CORE_MEETS_FROZEN_SUPPORT_GATE','meshes':rows,'components_in_selection_order':ordered,
 'selected':selected,'previous_centres':len(centres),'GO_sha256':sha(P/'GO.json'),'no_model_or_intensity_access':True,'gpu_spend_GBP':0,
 'limitations':['Conservative sphere, CT-box and all-origin context envelopes may reject otherwise disjoint exact footprints.','No physical-sheet or independent ink ground truth established.','Geometry support does not prove CT objects are present or execution renderer fidelity.']}
save(O/'RESULTS.json',result);print(json.dumps({'decision':result['decision'],'selected':selected}))
