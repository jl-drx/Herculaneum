#!/usr/bin/env python3
"""Reproduce archived render sampling at three points; never fit offsets or infer ink."""
from pathlib import Path
import json,hashlib,importlib.util,datetime,itertools
import numpy as np
import tifffile
import requests

LAB=Path('/workspace/herculaneum-lab')
ROOT=Path(__file__).resolve().parent
GEOM=LAB/'data/experiments/pherc0800-larger-geometry/20251028222030'
SURFACE=LAB/'data/experiments/pherc0800-larger1536'
REPORT=LAB/'reports/third-experiment'
COMMIT='530584554c031c982ba3aa243e1ea39c4bef9818'
URL='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0/'
CAP=16*1024*1024
spec=importlib.util.spec_from_file_location('fetch_zarr_chunk',LAB/'fetch_zarr_chunk.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def f32(v):return np.asarray(v,dtype=np.float32)
def fma32(a,b,c):return np.float32(float(a)*float(b)+float(c))
input_names=[GEOM/'mesh.meta.json',GEOM/'mesh.x.tif',GEOM/'mesh.y.tif',GEOM/'mesh.z.tif',GEOM/'full-surface-ct-coordinates.xyz.npy',SURFACE/'surface.npy',SURFACE/'provenance.json']
input_hashes={str(p):sha(p) for p in input_names}
export=json.load(open(ROOT/'target-segment-export-metadata.json'))
render=next(d for d in export['segment']['data'] if d['type']=='layers-zarr')['creation_info']['provenance']['parameters']
assert render['commit']==COMMIT and render['flip-normals']=='true'
assert render['group-idx']=='0' and render['scale']=='1' and render['num-slices']=='31'
assert render['rotate']=='0' and render['flip']=='-1' and render['composite']=='false'
assert all(render[k]=='' for k in ['affine-transform','affines','affine-matrix','scale-segmentation'])
assert all(render[k]=='0' for k in ['crop-x','crop-y','crop-width','crop-height'])
coords=np.stack([tifffile.imread(GEOM/f'mesh.{a}.tif') for a in 'xyz'],axis=-1)
scale=f32(json.load(open(GEOM/'mesh.meta.json'))['scale'])
full_map=np.load(GEOM/'full-surface-ct-coordinates.xyz.npy',mmap_mode='r')
surface=np.load(SURFACE/'surface.npy',mmap_mode='r')
assert surface.shape==(31,1536,1536)
assert sha(SURFACE/'surface.npy')==json.load(open(SURFACE/'provenance.json'))['array_sha256']
canvas_h,canvas_w=json.load(open(GEOM/'surface.level0.zarray.json'))['shape'][1:]
assert (canvas_h,canvas_w)==(2580,2580)
point_selection=json.load(open(ROOT/'point-selection.json'))
assert point_selection['points_surface_xy']==[[1280,1280],[1279,1280],[1280,1281]]

# Simulate archived float32 tile origin/halo arithmetic (unrotated, uncropped render).
def archived_base_normal(x,y):
    c0,r0=(x//128)*128,(y//128)*128
    center=f32([float(coords.shape[1])/2/float(scale[0]),float(coords.shape[0])/2/float(scale[1])])
    offset=f32([-0.5*(canvas_w-1)+c0,-0.5*(canvas_h-1)+r0])
    ul=f32(f32(offset+center)*scale)
    o=np.array([float(ul[k])-4*float(scale[k]) for k in [0,1]],dtype=np.float32)
    # Positive lround is floor+0.5. No pixel lies at a ties-to-even boundary here.
    gx=np.float32(o[0]+np.float32(x-c0+4)*scale[0]);gy=np.float32(o[1]+np.float32(y-r0+4)*scale[1])
    ix,iy=int(gx),int(gy);wx=np.float32(gx-ix);wy=np.float32(gy-iy)
    p00,p01,p10,p11=coords[iy,ix],coords[iy,ix+1],coords[iy+1,ix],coords[iy+1,ix+1]
    assert np.isfinite([p00,p01,p10,p11]).all() and (np.array([p00,p01,p10,p11])>0).all()
    base=f32(f32(f32(p00*np.float32(1-wx))+f32(p01*wx))*np.float32(1-wy)+f32(f32(p10*np.float32(1-wx))+f32(p11*wx))*wy)
    nx,ny=int(np.floor(float(gx)+.5)),int(np.floor(float(gy)+.5))
    xl,xr,yu,yd=coords[ny,nx-1],coords[ny,nx+1],coords[ny-1,nx],coords[ny+1,nx]
    assert np.isfinite([xl,xr,yu,yd]).all() and (np.array([xl,xr,yu,yd])>0).all()
    xv,yv=f32(xr-xl),f32(yd-yu)
    n=f32([xv[1]*yv[2]-xv[2]*yv[1],xv[2]*yv[0]-xv[0]*yv[2],xv[0]*yv[1]-xv[1]*yv[0]])
    len2=np.float32(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);assert np.isfinite(len2) and len2>0
    n=f32(n*np.float32(1/np.sqrt(len2)))
    n=f32(-n) # --flip-normals=true is recorded, not fitted.
    len2=np.float32(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);n=f32(n/np.sqrt(len2))
    assert np.linalg.norm(base-full_map[y,x])<.01
    positions=np.stack([f32([fma32(n[k],off,base[k]) for k in [0,1,2]]) for off in range(-15,16)])
    return {'surface_xy':[x,y],'grid_xy':[float(gx),float(gy)],'normal_grid_xy':[nx,ny],'base_ct_xyz':base.tolist(),'cached_map_ct_xyz':full_map[y,x].tolist(),'cached_vs_archived_base_max_abs_error_voxels':float(np.abs(base-full_map[y,x]).max()),'normal_xyz_flipped':n.tolist(),'positions':positions,'surface_local_roi_xy':[x-512,y-512],'observed_profile':surface[:,y-512,x-512].astype(np.uint8)}

points=[archived_base_normal(x,y) for x,y in point_selection['points_surface_xy']]
raw_meta,meta_prov=mod.fetch(URL+'.zarray',256*1024)
metadata=json.loads(raw_meta)
assert metadata['shape']==[24298,9867,9867] and metadata['chunks']==[128]*3 and metadata['dtype']=='|u1'
assert metadata['compressor'] is None and metadata.get('filters') is None and metadata['fill_value']==0
(ROOT/'ct.level0.zarray.json').write_bytes(raw_meta)
chunk_indices=set()
for p in points:
 for xyz in p['positions']:
  zyx=xyz[::-1];base=np.floor(zyx).astype(int)
  for delta in itertools.product([0,1],repeat=3):
   v=base+np.array(delta)
   assert (v>=0).all() and (v<metadata['shape']).all()
   chunk_indices.add(tuple(map(int,v//128)))
chunk_bytes=128**3
assert len(chunk_indices)*chunk_bytes<=CAP
plan={'export_commit':COMMIT,'sample_points_surface_xy':[p['surface_xy'] for p in points],'requested_chunk_indices_zyx':[list(x) for x in sorted(chunk_indices)],'forecast_native_payload_bytes':len(chunk_indices)*chunk_bytes,'native_budget_bytes':CAP,'known_offsets_voxels':list(range(-15,16)),'normal_sign':'Recorded flip-normals=true','no_offset_search':True}
(ROOT/'native-ct-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Verified exact export parameters; bounded native chunk plan',plan,flush=True)
arrays={};sources=[];downloaded=0
for index in sorted(chunk_indices):
 key='/'.join(map(str,index));fn='ct-chunk-'+ '-'.join(map(str,index))+'.bin'
 url=URL+key
 try:
  raw,prov=mod.fetch(url,chunk_bytes)
  if len(raw)!=chunk_bytes:raise ValueError('Unexpected native chunk byte length')
  downloaded+=len(raw)
  assert downloaded<=CAP
  path=ROOT/fn
  if path.exists():assert path.read_bytes()==raw,'Existing retained chunk differs'
  else:path.write_bytes(raw)
  arrays[index]=np.frombuffer(raw,dtype=np.uint8).reshape((128,)*3)
  prov.update(file=fn,chunk_index_zyx=list(index),missing=False)
 except requests.HTTPError as e:
  if not getattr(e,'zarr_missing_key',False):raise
  arrays[index]=np.zeros((128,)*3,dtype=np.uint8)
  prov={'url':url,'chunk_index_zyx':list(index),'missing':True,'http_status':404,'diagnosis':'S3 NoSuchKey','reconstruction':'Declared Zarr fill_value=0; not a downloaded chunk','decoded_size_bytes':chunk_bytes}
 sources.append(prov)

def at(z,y,x):
 index=(z//128,y//128,x//128)
 return np.float32(arrays[index][z%128,y%128,x%128])
def trilinear(xyz):
 x,y,z=map(np.float32,xyz);iz,iy,ix=int(z),int(y),int(x)
 c000,c100=at(iz,iy,ix),at(iz+1,iy,ix)
 c010,c110=at(iz,iy+1,ix),at(iz+1,iy+1,ix)
 c001,c101=at(iz,iy,ix+1),at(iz+1,iy,ix+1)
 c011,c111=at(iz,iy+1,ix+1),at(iz+1,iy+1,ix+1)
 fz,fy,fx=np.float32(z-iz),np.float32(y-iy),np.float32(x-ix)
 c00=fma32(fx,np.float32(c001-c000),c000);c01=fma32(fx,np.float32(c011-c010),c010)
 c10=fma32(fx,np.float32(c101-c100),c100);c11=fma32(fx,np.float32(c111-c110),c110)
 c0=fma32(fy,np.float32(c01-c00),c00);c1=fma32(fy,np.float32(c11-c10),c10)
 return fma32(fz,np.float32(c1-c0),c0)
for p in points:
 pred=np.array([trilinear(v) for v in p.pop('positions')],dtype=np.float32)
 pred_uint=np.clip(pred,0,255).astype(np.uint8) # Archived uint8 T(v) truncates, no round.
 obs=p.pop('observed_profile')
 p.update(observed_profile_uint8=obs.tolist(),source_ct_profile_float=pred.tolist(),archived_forward_prediction_uint8=pred_uint.tolist(),forward={'mean_abs_error':float(np.abs(obs.astype(float)-pred_uint).mean()),'max_abs_error':int(np.abs(obs.astype(int)-pred_uint.astype(int)).max()),'exact_matches':int((obs==pred_uint).sum()),'within_one_intensity_count':int((np.abs(obs.astype(int)-pred_uint.astype(int))<=1).sum())},reverse_order_diagnostic={'mean_abs_error':float(np.abs(obs.astype(float)-pred_uint[::-1]).mean()),'max_abs_error':int(np.abs(obs.astype(int)-pred_uint[::-1].astype(int)).max()),'exact_matches':int((obs==pred_uint[::-1]).sum())})
all_one=all(p['forward']['within_one_intensity_count']==31 for p in points)
result={'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sample_id':'PHerc0800','target_volume_id':'20250521135224','segment_id':'20251028222030','scientific_role':'Data integrity/source-render consistency diagnostic only, no model inference or letter reading.','export_provenance':render,'source_commit':COMMIT,'export_commit_retrieved_from_full_metadata':True,'metadata_provenance':json.load(open(ROOT/'metadata-provenance.json')),'full_metadata_context_path':str(ROOT/'target-segment-export-metadata.json'),'native_array_metadata':metadata,'native_array_metadata_source':meta_prov,'native_chunk_sources':sources,'downloaded_native_chunk_bytes':downloaded,'native_chunk_decoded_bytes':len(arrays)*chunk_bytes,'max_native_budget_bytes':CAP,'fixed_depth_offsets_voxels':list(range(-15,16)),'sampling_convention':'Recorded31slice export: flipped nearest cached grid normal; centered layer offsets-15..15; source bilinear pixel coordinates with halfpixel; trilinear nativeCT interpolation; clip0..255 then truncate uint8. No fitted offsets/transforms. Reverse control reverses the fixed31samples.','implementation_precision_note':'Python emulates float32 normal/position arithmetic and float32 fused trilinear operations. Compiled ARM floating-point contraction can introduce small quantization differences; no parameter fitting is performed.','points':points,'all_profiles_within_one_intensity_count':all_one,'status':'supports_recorded_source_render_mapping_at_tested_points' if all_one else 'source_render_mapping_not_reproduced_at_tested_points','input_files_sha256':input_hashes,'point_selection':point_selection,'point_selection_sha256':sha(ROOT/'point-selection.json'),'sampling_helper_sha256':sha(__file__),'adapted_from_helper':{'path':str(LAB/'data/experiments/pherc0800-native-ct-consistency/check_native_ct.py'),'sha256':sha(LAB/'data/experiments/pherc0800-native-ct-consistency/check_native_ct.py')},'limitations':['Three selected31layer profiles cannot validate all surface pixels or every render partition.','A reproducible intensity profile cannot prove ink presence, text legibility or absence of prediction hallucinations.','The native public volume is compared to the archived export of an internal volume path; identity is tested at these points, not assumed globally.']}
REPORT.mkdir(parents=True,exist_ok=True)
(REPORT/'native-ct-consistency.json').write_text(json.dumps(result,indent=2)+'\n')
assert input_hashes=={str(p):sha(p) for p in input_names},'Inputs changed during check'
print(json.dumps({'status':result['status'],'downloaded_native_chunk_bytes':downloaded,'points':points},indent=2))
