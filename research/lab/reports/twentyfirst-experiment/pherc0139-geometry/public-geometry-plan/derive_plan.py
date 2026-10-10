"""Synthetic index arithmetic and retained metadata only; no coordinate payloads."""
import hashlib,json,pathlib,math
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;P=ROOT.parent
F=np.float32
def rec(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def require(x,msg):
 if not x:raise ValueError(msg)
def axis_dependencies(mesh_count,canvas_count,scale,first,length,tile_size=128):
 require(first%tile_size==0 and length%tile_size==0,'fixed context tile alignment');s=F(scale);center=F(float(mesh_count)/2.0/float(s));positions=[];positions_fused=[]
 for start in range(first,first+length,tile_size):
  # computeCanvasOrigin + crop(0) + tile-start, exactly staged float32.
  origin=F(F(-.5)*F(F(canvas_count)-F(1)));offset=F(F(origin+F(0))+F(start))
  ul=F(F(F(offset/F(1))+center)*s)
  ox=F(float(ul)-4.0*float(s))
  for j in range(tile_size):
   dx=F(j+4);mul=F(dx*s)
   positions.append(F(ox+mul));positions_fused.append(F(float(ox)+float(dx)*float(s)))
 a=np.asarray(positions,dtype=np.float32);b=np.asarray(positions_fused,dtype=np.float32);require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)),'synthetic mapping nonfinite')
 allpos=np.concatenate([a,b]);require(np.all(allpos>=0) and np.all(allpos<mesh_count-1),'synthetic raster dependency at grid border')
 bilinear0=np.floor(allpos.astype(np.float64)).astype(np.int64);bilinear1=bilinear0+1
 nearest=np.floor(allpos.astype(np.float64)+.5).astype(np.int64) # nonnegative lround, ties away fromzero
 lo=min(int(bilinear0.min()),int(nearest.min())-1);hi=max(int(bilinear1.max()),int(nearest.max())+1)
 require(lo>=0 and hi<mesh_count,'dependency exceeds mesh')
 return {'fine_first':first,'fine_length':length,'tile_size':tile_size,'mesh_count':mesh_count,'canvas_count':canvas_count,'float32_scale':float(s),'float32_center':float(center),'coordinate_min':float(allpos.min()),'coordinate_max':float(allpos.max()),'bilinear_grid_min':int(bilinear0.min()),'bilinear_grid_max':int(bilinear1.max()),'nearest_grid_min':int(nearest.min()),'nearest_grid_max':int(nearest.max()),'dependency_inclusive':[lo,hi],'dependency_count':hi-lo+1,'fused_vs_separate_position_max_abs':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),'fused_vs_separate_nearest_indices_equal':bool(np.array_equal(np.floor(a.astype(np.float64)+.5),np.floor(b.astype(np.float64)+.5))),'fused_vs_separate_bilinear_indices_equal':bool(np.array_equal(np.floor(a),np.floor(b))),'construction':'Union of separately rounded and one-FMA multiply-add position paths; bilinearfloor/+1 and nearestlround/+/-1. Index arrays are synthetic, no coordinate samples.'}
def main():
 meta=json.loads((P/'text-run/transformed_mesh_meta.body').read_text());objects=[json.loads((P/'structural/run'/f'{c}.tif.contract.json').read_text()) for c in 'xyz'];require(all(d['width']==1444 and d['height']==1404 and d['byte_order']=='<' and d['bits_per_sample']==32 and d['sample_format']==3 and d['compression']==1 and d['storage_offsets']==[8] and d['storage_byte_counts']==[8109504] for d in objects),'observed structural contract')
 contexts=[];requests=[]
 for name,box in [('pherc0139-w016-rank01',[1792,4864,384,384]),('pherc0139-w016-rank02',[3584,5120,384,384])]:
  x,y,w,h=box;fine=[4*x,4*y,4*w,4*h];dx=axis_dependencies(1444,28880,meta['scale'][0],fine[0],fine[2]);dy=axis_dependencies(1404,28080,meta['scale'][1],fine[1],fine[3]);x0,x1=dx['dependency_inclusive'];y0,y1=dy['dependency_inclusive'];mw=x1-x0+1;mh=y1-y0+1
  ranges=[]
  for obj in objects:
   first=8+4*(y0*1444+x0);end=8+4*(y1*1444+x1+1);require(8<=first<end<=8109512,'payload range extent');length=end-first
   row={'id':name+'-'+obj['source']['name'],'context':name,'source':obj['source'],'range_inclusive':[first,end-1],'expected_body_bytes':length,'reserved_body_bytes':length+1,'useful_coordinate_bytes':mw*mh*4,'uninterpreted_gap_bytes':length-mw*mh*4,'row_stride_bytes':1444*4,'window_xywh':[x0,y0,mw,mh],'decode_slices_relative':'For each j=0..height-1 decode raw[j*5776:j*5776+width*4] only; preserve all intervening bytes opaque.','no_payload_read_or_requested':True};requests.append(row);ranges.append(row['id'])
  contexts.append({'id':name,'level2_bbox_xywh':box,'fine_raster_bbox_xywh':fine,'axis_x':dx,'axis_y':dy,'mesh_window_xywh':[x0,y0,mw,mh],'requests':ranges,'selected_source_depth_half_open':[21,89],'normal_offsets_inclusive':[-33,34],'all_depth_count':68,'fine_xy_contributors_per_level2_output':16})
 plan={'schema_version':1,'status':'PROPOSAL_ONLY_NO_ACQUISITION_AUTHORIZED','ready':False,'scope':'Current public mesh computational geometry for exact two frozen PHerc0139 trainingcontexts only; no CT/mask/labels/models/weights','contexts':contexts,'requests':requests,'maximum_attempts':6,'maximum_total_reserved_response_body_bytes':4*1024*1024,'derived_total_expected_body_bytes':sum(x['expected_body_bytes'] for x in requests),'derived_total_reserved_body_bytes':sum(x['reserved_body_bytes'] for x in requests),'derived_total_useful_coordinate_bytes':sum(x['useful_coordinate_bytes'] for x in requests),'additional_header_requests':0,'header_policy':'Reuse pinned successful structural contracts and exact inventory ETags/sizes. Changed ETag/size aborts; no automatic headerrefresh.','per_request_timeout_seconds':30,'external_wall_seconds':240,'transport':['Exactly six sequential HTTPS single Range GETs; no multipart, retries, redirects or fullbodyfallback.','Every request sends pinned strong If-Match, requires exact206 Content-Range/total,identityencoding and exact expected length.','Reserve full requestedcap before opening; persist partial/failure status and retain full reservation on unknown failures; stop first failure.','Do not interpret bytes outside listed mesh-window row slices; no coordinate-based support expansion or new targets.'],'decode_budget':{'maximum_peak_rss_bytes':536870912,'maximum_output_bytes':33554432,'processing':'Two contexts sequentially; one128x128 fine-raster tile at a time; reduce exact4x4fine validity into384x384 boolean outputs. No full109-plane or full1536x1536x68 coordinatevolume.'},'reconstruction_status':'Definitions/proofs/fixtures in RECONSTRUCTION_CONTRACT.md and synthetic_checks.py; implementation and independent review required before acquisition/reconstruction.','output_claim':'Reconstructed current public mesh geometric/computational support under explicit conservative finite/stencil/native-frame bounds. Not historical private-render validity, observed nativeCT/material support, ink or correctness.','fixed_study_boundary':'Never changes already running S/V allocation-study masks, objective, labels or decisions.','input_pins':[rec(P/'structural/run/result.json'),rec(P/'structural/STRUCTURAL_MANIFEST.json'),rec(P/'text-run/result.json'),rec(P/'text-run/transformed_mesh_meta.body')]+[rec(P/'structural/run'/f'{c}.tif.contract.json') for c in 'xyz']+[rec(P/'text-run'/f'{n}.body') for n in ['renderer','loader_raster','normals','ct_sampling','pyramid_implementation']]}
 plan['native_frame']=json.loads((ROOT/'NATIVE_FRAME_METADATA.json').read_text());plan['input_pins'].append(rec(ROOT/'NATIVE_FRAME_METADATA.json'));require(plan['derived_total_reserved_body_bytes']<=4194304,'payload cap');(ROOT/'COORDINATE_QUERY_PLAN.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');print(json.dumps({'plan':rec(ROOT/'COORDINATE_QUERY_PLAN.json'),'windows':[c['mesh_window_xywh'] for c in contexts],'body':plan['derived_total_expected_body_bytes'],'reserved':plan['derived_total_reserved_body_bytes'],'useful':plan['derived_total_useful_coordinate_bytes']}))
if __name__=='__main__':main()
