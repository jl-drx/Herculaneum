import json,urllib.request,hashlib,datetime,concurrent.futures,io
from pathlib import Path
import numpy as np
import tifffile
ROOT=Path('/workspace/herculaneum-lab/data/experiments/pherc0800-surface-geometry')
ROOT.mkdir(parents=True,exist_ok=True)
inv=json.load(open('/workspace/herculaneum-lab/reports/live-target-inventory.json'))
seg=next(s for s in inv['segments'] if s['segment_id']=='20251029010146')
mesh=seg['mesh_url'];surface=seg['surface_url']
manifest={'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sample_id':'PHerc0800','eligible_volume_id':'20250521135224','segment_id':seg['segment_id'],'segment_long_id':seg['long_id'],'license':inv['license'],'interpretation':'Surface/geometry inspection only; no ink model, transcription or translation.','artifacts':[]}
qs=[('mesh.meta.json',mesh+'meta.json',16384),('mesh.x.tif',mesh+'x.tif',100000),('mesh.y.tif',mesh+'y.tif',100000),('mesh.z.tif',mesh+'z.tif',100000),('surface.zattrs.json',surface+'.zattrs',32768),('surface.zgroup.json',surface+'.zgroup',16384),('surface.level0.zarray.json',surface+'0/.zarray',16384),('surface.level3.zarray.json',surface+'3/.zarray',16384),('surface.level3.chunk',surface+'3/0/0/0',600000)]
expected={r['url']:r.get('headers',{}).get('etag') for r in seg['head_probes']}
expected.update({r['url']:r.get('headers',{}).get('etag') for r in inv['extra_head_probes']})
def fetch(q):
 name,url,cap=q
 headers={'Accept-Encoding':'identity'}
 if expected.get(url):headers['If-Match']=expected[url]
 with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=25) as r:
  payload=r.read(cap+1)
  if len(payload)>cap:raise ValueError(f'{name}: bounded size cap exceeded')
  if r.headers.get('Content-Length') and len(payload)!=int(r.headers['Content-Length']):raise ValueError(f'{name}: Content-Length mismatch')
  if expected.get(url) and r.headers.get('ETag')!=expected[url]:raise ValueError(f'{name}: ETag changed since inventory')
  row={'file':name,'source_url':url,'http_status':r.status,'byte_count':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'http_headers':dict(r.headers.items()),'prior_head_etag':expected.get(url),'prior_etag_matched':True if expected.get(url) else None}
 p=ROOT/name
 if p.exists():
  if p.read_bytes()!=payload:raise ValueError(f'{name}: existing local artifact differs; refusing overwrite')
 else:p.write_bytes(payload)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:manifest['artifacts']=list(ex.map(fetch,qs))
meta=json.load(open(ROOT/'mesh.meta.json'))
arrmeta=json.load(open(ROOT/'surface.level3.zarray.json'))
attrs=json.load(open(ROOT/'surface.zattrs.json'))
payload=(ROOT/'surface.level3.chunk').read_bytes()
dtype=np.dtype(arrmeta['dtype'])
chunk_shape=tuple(arrmeta['chunks']);logical_shape=tuple(arrmeta['shape'])
if arrmeta.get('compressor') is not None or arrmeta.get('filters') is not None:raise ValueError('Decoder only valid for uncompressed/unfiltered source')
items=len(payload)//dtype.itemsize
if len(payload)%dtype.itemsize:raise ValueError('Chunk byte alignment invalid')
if items==int(np.prod(chunk_shape)):
 overview=np.frombuffer(payload,dtype=dtype).reshape(chunk_shape,order=arrmeta['order'])
 decoded_shape=chunk_shape
elif items==31*128*128 and logical_shape==(31,123,123):
 overview=np.frombuffer(payload,dtype=dtype).reshape((31,128,128),order=arrmeta['order'])
 decoded_shape=(31,128,128)
else:raise ValueError(f'Unexpected chunk payload: {items} items for {chunk_shape}')
overview=np.ascontiguousarray(overview[tuple(slice(0,n) for n in logical_shape)])
np.save(ROOT/'overview.npy',overview)
coords=np.stack([tifffile.imread(ROOT/f'mesh.{k}.tif') for k in 'xyz'],axis=-1)
assert coords.shape==(49,49,3) and coords.dtype==np.float32
valid=np.isfinite(coords).all(axis=-1)&(coords!=-1).all(axis=-1)
assert valid.any()
valid_points=coords[valid].astype(np.float64)
computed_bbox=np.stack([valid_points.min(axis=0),valid_points.max(axis=0)])
reported_bbox=np.array(meta['bbox'],dtype=np.float64)
assert np.allclose(computed_bbox,reported_bbox,rtol=0,atol=0.002)
scale_xy=np.array(meta['scale'],dtype=np.float64)
assert np.allclose(scale_xy,[.05,.05])
assert tuple(np.rint(np.array(coords.shape[:2])/scale_xy[::-1]).astype(int))==(980,980)
# vc_render_tifxyz computeCanvasOrigin=-0.5*(full_size-1), QuadSurface center=stored_size/(2*scale).
# Thus native render pixel center (x,y) maps grid ((x+0.5)*scale_x,(y+0.5)*scale_y).
def bilinear(y,x):
 y,x=np.broadcast_arrays(np.asarray(y,dtype=np.float64),np.asarray(x,dtype=np.float64))
 gy=(y+.5)*scale_xy[1];gx=(x+.5)*scale_xy[0]
 iy=np.floor(gy).astype(int);ix=np.floor(gx).astype(int)
 inside=(iy>=0)&(ix>=0)&(iy+1<coords.shape[0])&(ix+1<coords.shape[1])
 iyc=np.clip(iy,0,coords.shape[0]-2);ixc=np.clip(ix,0,coords.shape[1]-2)
 fy=gy-iy;fx=gx-ix
 v=inside&valid[iyc,ixc]&valid[iyc+1,ixc]&valid[iyc,ixc+1]&valid[iyc+1,ixc+1]
 out=((1-fy)*(1-fx))[...,None]*coords[iyc,ixc]+((1-fy)*fx)[...,None]*coords[iyc,ixc+1]+(fy*(1-fx))[...,None]*coords[iyc+1,ixc]+(fy*fx)[...,None]*coords[iyc+1,ixc+1]
 out=np.where(v[...,None],out,np.nan)
 return out,v
Y,X=np.meshgrid(np.arange(256,512),np.arange(256,512),indexing='ij')
roi,roi_valid=bilinear(Y,X)
assert roi_valid.any()
pts=roi[roi_valid]
roi_bbox=np.stack([pts.min(axis=0),pts.max(axis=0)])
center,center_valid=bilinear(383.5,383.5)
assert center_valid
sample384,sample384valid=bilinear(384,384)
ctshape=np.array([24298,9867,9867])
assert ((valid_points>=0)&(valid_points<ctshape[::-1])).all()
summary={'sample_id':'PHerc0800','volume_id':'20250521135224','segment_id':seg['segment_id'],'array_metadata_shape_zyx':logical_shape,'array_metadata_chunk_shape_zyx':chunk_shape,'decoded_wire_chunk_shape_zyx':decoded_shape,'overview_shape_zyx':overview.shape,'padding_clipped':decoded_shape!=logical_shape,'overview_voxel_size_um_zyx':[8.64,69.12,69.12],'overview_statistics':{'min':int(overview.min()),'max':int(overview.max()),'mean':float(overview.mean()),'nonzero_fraction':float(np.count_nonzero(overview)/overview.size)},'mesh_coordinate_shape_yx':coords.shape[:2],'mesh_coordinate_dtype':str(coords.dtype),'mesh_valid_vertices':int(valid.sum()),'mesh_total_vertices':int(valid.size),'mesh_valid_vertex_fraction':float(valid.mean()),'mesh_bbox_computed_xyz':computed_bbox.tolist(),'mesh_bbox_reported_xyz':reported_bbox.tolist(),'mesh_bbox_max_abs_difference':float(np.abs(computed_bbox-reported_bbox).max()),'mesh_scale_xy':scale_xy.tolist(),'native_render_shape_yx':[980,980],'native_ct_shape_zyx':ctshape.tolist(),'mapping_convention':'Native render pixel centers map to mesh grid: grid_x=(x+0.5)*scale_x; grid_y=(y+0.5)*scale_y. Bilinear interpolation requires all four finite/non-sentinel source vertices. XYZ coordinates are native target CT voxel indices.','mapping_source_commit':'e0bbb8b40a2db58b1d71864f286eb85717e59e64','mapping_sources':{'canvas_origin':'https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/volume-cartographer/apps/src/vc_render_tifxyz.cpp#L280','surface_center':'https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/volume-cartographer/core/src/QuadSurface.cpp#L282','generation_grid_transform':'https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/volume-cartographer/core/src/QuadSurface.cpp#L1644'},'native_roi_yx_half_open':[[256,512],[256,512]],'native_roi_shape_yx':[256,256],'native_roi_valid_bilinear_pixels':int(roi_valid.sum()),'native_roi_total_pixels':int(roi_valid.size),'native_roi_valid_fraction':float(roi_valid.mean()),'native_roi_center_pixel_yx':[383.5,383.5],'native_roi_center_grid_yx':[(383.5+.5)*.05]*2,'native_roi_center_ct_xyz':center.tolist(),'native_roi_integer_center_sample_yx':[384,384],'native_roi_integer_center_sample_ct_xyz':sample384.tolist(),'native_roi_ct_surface_bbox_xyz':roi_bbox.tolist(),'conservative_ct_bbox_with_16_voxel_margin_xyz':np.stack([np.floor(roi_bbox[0])-16,np.ceil(roi_bbox[1])+16]).tolist(),'margin_note':'Surface geometry bbox only. The conservative ±16 voxel margin can cover a centered ±15-voxel depth stack plus rounding, but the archived render zero-depth orientation/offset is not explicitly recorded in surface attrs and must be checked against source CT.','native_roi_nominal_planar_size_mm_yx':[256*.00864]*2,'geometry_validation_limits':['Original September2026 export renderer commit not recorded; mapping uses current pinned maintained C++ renderer convention.','Bilinear numeric mapping does not infer ink and does not prove flattening distortion absent.','Validity uses finite XYZ and no -1 sentinel, matching C++ point validity; no sidecar mask is advertised in live projection.']}
(ROOT/'geometry-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
np.save(ROOT/'native-roi-ct-coordinates.xyz.npy',roi.astype(np.float32))
np.save(ROOT/'native-roi-valid-mask.npy',roi_valid)
manifest['derived_artifacts']=[]
for fn in ['overview.npy','geometry-validation.json','native-roi-ct-coordinates.xyz.npy','native-roi-valid-mask.npy']:
 raw=(ROOT/fn).read_bytes();manifest['derived_artifacts'].append({'file':fn,'byte_count':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'processing':'Deterministic metadata-driven byte decoding or bilinear geometry mapping; details in geometry-validation.json.'})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
