"""Finite native XYZ context overlap audit; no ink pixels or CT reads."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.util,json,math
import numpy as np
import tifffile
P=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def need(x,m):
 if not x:raise ValueError(m)
def load_helper():
 spec=importlib.util.spec_from_file_location('retained_native_geometry',P/'dependencies/native_geometry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def alternative_base(mesh,xs,ys,canvas,scale):
 # Full precision equivalent of canvas/mesh center convention. Actual staged
 # FP32 output is separately retained; union of extrema is used conservatively.
 h,w=mesh.shape[:2]
 qx=(np.asarray(xs,dtype=np.float64)-.5*(canvas[0]-1)+w/(2*scale))*scale
 qy=(np.asarray(ys,dtype=np.float64)-.5*(canvas[1]-1)+h/(2*scale))*scale
 ix=np.floor(qx).astype(int);iy=np.floor(qy).astype(int)
 need(ix.min()>=0 and ix.max()+1<w and iy.min()>=0 and iy.max()+1<h,'Alternative interpolation bounds')
 xx=(qx-ix)[None,:,None];yy=(qy-iy)[:,None,None]
 corners=[mesh[iy[:,None]+dy,ix[None,:]+dx].astype(np.float64) for dy,dx in [(0,0),(0,1),(1,0),(1,1)]]
 valid=np.logical_and.reduce([np.isfinite(v).all(-1)&(v[...,2]>0)&(v[...,0]!=-1) for v in corners])
 value=(corners[0]*(1-xx)+corners[1]*xx)*(1-yy)+(corners[2]*(1-xx)+corners[3]*xx)*yy
 return value,valid

def intersects(a,b):
 return bool(np.all(np.asarray(a[0])<=np.asarray(b[1])) and np.all(np.asarray(b[0])<=np.asarray(a[1])))

def boxes(base,alt,normal,valid):
 if not valid.any():return None,None,None
 both=np.concatenate([base[valid],alt[valid]],axis=0)
 minimum,maximum=both.min(0),both.max(0)
 conservative=[(minimum-14.51).tolist(),(maximum+14.51).tolist()]
 rays=np.concatenate([base[valid]-13.5*normal[valid],base[valid]+13.5*normal[valid]],axis=0)
 exact=[(rays.min(0)-1.01).tolist(),(rays.max(0)+1.01).tolist()]
 need(np.all(np.asarray(exact[0])>=np.asarray(conservative[0])) and np.all(np.asarray(exact[1])<=np.asarray(conservative[1])),'Ray box inside conservative box')
 return conservative,exact,[minimum.tolist(),maximum.tolist()]

def fold_rows(rows):
 folds=[]
 for held in ['w035','w039','w044']:
  evaluation=[r for r in rows if r['winding']==held];training=[r for r in rows if r['winding']!=held]
  kept=[];rejected=[]
  for r in training:
   matches=[e['core_id'] for e in evaluation if r['bbox_conservative_xyz'] is not None and e['bbox_conservative_xyz'] is not None and intersects(r['bbox_conservative_xyz'],e['bbox_conservative_xyz'])]
   reasons=[]
   if not r['geometry_qualified']:reasons.append('unqualified_training_geometry')
   if matches:reasons.append('conservative_box_intersects_heldout_context')
   if reasons:rejected.append(dict(core_id=r['core_id'],winding=r['winding'],reasons=reasons,intersecting_evaluation_ids=matches))
   else:kept.append(r)
  invalid=[r['core_id'] for r in evaluation if not r['geometry_qualified']]
  folds.append(dict(held_out=held,original_evaluation_core_ids=[r['core_id'] for r in evaluation],evaluation_cores=len(evaluation),
   unqualified_evaluation_core_ids=invalid,fold_geometry_admitted=(not invalid and bool(kept)),
   original_training_cores=len(training),retained_training_cores=len(kept),retained_training_core_ids=[r['core_id'] for r in kept],
   retained_training_supervised_pixels=sum(r['supervised_pixels'] for r in kept),
   retained_training_by_winding={w:dict(cores=sum(r['winding']==w for r in kept),supervised_pixels=sum(r['supervised_pixels'] for r in kept if r['winding']==w)) for w in ['w035','w039','w044'] if w!=held},
   rejected_training_contexts=rejected))
 return folds

def synthetic():
 m=load_helper();yy,xx=np.mgrid[:30,:30];mesh=np.stack([xx*20,yy*20,100+xx*2+yy*3],-1).astype(np.float32)
 base,normal,valid=m.render(mesh,m.normals(mesh),np.arange(80,160),np.arange(100,180),(600,600))
 alt,av=alternative_base(mesh,np.arange(80,160),np.arange(100,180),(600,600),float(m.S));valid&=av
 need(valid.all(),'Tilted plane valid');need(np.max(np.abs(np.linalg.norm(normal,axis=-1)-1))<1e-5,'Unit tilted normals')
 need(np.max(np.abs(base[...,0]-(np.arange(80,160)[None,:]+.5)))<.0001,'Canvas half pixel X')
 bb,rb,_=boxes(base,alt,normal,valid)
 for depth in np.arange(28)-13.5:
  points=base+depth*normal
  need(np.all(points[valid]-1>=np.asarray(bb[0])) and np.all(points[valid]+1<=np.asarray(bb[1])),'All28 positions and trilinear neighbor enclosure')
 flat=mesh.copy();flat[...,2]=100;base,n,v=m.render(flat,m.normals(flat),np.arange(80,160),np.arange(100,180),(600,600));need(np.max(np.abs(n[...,2]+1))<1e-7,'Flat flipped normal')
 need(intersects([[0,0,0],[1,1,1]],[[1,1,1],[2,2,2]]),'Closed boundary intersection');need(not intersects([[0,0,0],[1,1,1]],[[1.001,1,1],[2,2,2]]),'Disjoint box')
 fixture=[dict(core_id='a',winding='w035',geometry_qualified=True,bbox_conservative_xyz=[[0,0,0],[1,1,1]],supervised_pixels=5),dict(core_id='b',winding='w039',geometry_qualified=True,bbox_conservative_xyz=[[1,1,1],[2,2,2]],supervised_pixels=3),dict(core_id='c',winding='w044',geometry_qualified=True,bbox_conservative_xyz=[[3,3,3],[4,4,4]],supervised_pixels=7)]
 f=fold_rows(fixture)[0];need(f['retained_training_core_ids']==['c'] and f['rejected_training_contexts'][0]['core_id']=='b','Fold exclusion logic')
 save(P/'SYNTHETIC_CHECKS.json',dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),checks=['tilted_and_flat_planes','canvas_half_pixel','normal_flip','all28_rays_and_trilinear_support_enclosed','closed_boundary_and_disjoint_boxes','leave_one_winding_out_exclusion'],code_sha256=sha(Path(__file__))))

def main():
 need(not (P/'RESULTS.json').exists(),'Refuse existing result');synthetic()
 plan=json.loads((P/'SOURCE_PLAN.json').read_text());need(sha(P/'PROTOCOL.md')==plan['protocol_sha256'],'Frozen protocol')
 allpaths=[P/'PROTOCOL.md',P/'COHORT_COUNT_ERRATUM.md',P/'SOURCE_PLAN.json',P/'PAYLOAD_FREEZE.json',P/'METADATA_LEDGER.json',P/'COORDINATE_LEDGER.json',Path(__file__),P/'SYNTHETIC_CHECKS.json']+list((P/'dependencies').glob('*'))+list((P/'responses').glob('*'))
 for w in ['w035','w039','w044']:allpaths.append(P.parent/f'{w}-eligible-cores.csv')
 before=[dict(relative=str(p.relative_to(P)) if p.is_relative_to(P) else '../'+p.name,sha256=sha(p),size_bytes=p.stat().st_size) for p in allpaths if p.is_file()]
 save(P/'COMPUTE_FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files=before))
 for r in json.loads((P/'COORDINATE_LEDGER.json').read_text()):need(sha(P/'responses'/r['name'])==r['sha256'],'Acquired mesh checksum')
 helper=load_helper();rows=[];site_reports=[]
 for site in plan['sites']:
  w=site['winding'];attributes=site['native_attrs'];params=site['render']['creation_info']['provenance']['parameters'];meshparams=site['mesh']['parameters'];meta=json.loads((P/'responses'/f'{w}-meta.json').read_text())
  expected={'commit':'530584554c031c982ba3aa243e1ea39c4bef9818','scale':'1','group-idx':'0','num-slices':'28','rotate':'0','flip':'-1','flip-normals':'true','crop-x':'0','crop-y':'0','crop-width':'0','crop-height':'0','affine-matrix':'','affine-transform':'','affines':'','scale-segmentation':''}
  need(all(params.get(k)==v for k,v in expected.items()),'Native renderer contract')
  need(meshparams['target_volume']=='20250728140407' and attributes['num_slices']==28 and attributes['slice_step']==1 and attributes['source_group']==0,'Common native frame and full depth')
  need(meta['scale']==[float(helper.S)]*2,'Exact mesh scale')
  mesh_arrays=[tifffile.imread(P/'responses'/f'{w}-{axis}.tif') for axis in 'xyz'];need(all(a.dtype==np.float32 and a.shape==mesh_arrays[0].shape for a in mesh_arrays),'Coordinate shape and dtype')
  mesh=np.stack(mesh_arrays,-1);cache=helper.normals(mesh);canvas=attributes['canvas_size'];h,nw=mesh.shape[:2]
  need(canvas==[math.ceil(nw/float(helper.S)),math.ceil(h/float(helper.S))],'Canvas from source mesh scale')
  nvalid=cache[2];normerror=float(np.max(np.abs(np.linalg.norm(cache[1][nvalid].astype(float),axis=-1)-1))) if nvalid.any() else None
  need(normerror is not None and normerror<1e-5,'Source normals are unit within tolerance')
  cohort=list(csv.DictReader((P.parent/f'{w}-eligible-cores.csv').open()));need(all(r['eligible']=='True' and r['complete_context']=='True' for r in cohort),'Original mask-only frozen cohort')
  site_reports.append(dict(winding=w,public_segment=site['public_segment'],mesh_shape_yxc=list(mesh.shape),canvas_wh=canvas,source_volume='20250728140407',scale=meta['scale'],normal_max_norm_error=normerror,cores=len(cohort)))
  for core in cohort:
   x,y=int(core['x']),int(core['y']);xs=np.arange(x-64,x+192);ys=np.arange(y-64,y+192)
   need(xs[0]>=0 and ys[0]>=0 and xs[-1]<canvas[0] and ys[-1]<canvas[1],'Complete frozen context canvas')
   try:
    base,normal,valid=helper.render(mesh,cache,xs,ys,canvas);alt,av=alternative_base(mesh,xs,ys,canvas,float(helper.S));valid&=av
    bbox,raybox,basebox=boxes(base,alt,normal,valid);qualified=bool(valid.all());err=None
    delta=float(np.max(np.abs(base[valid]-alt[valid]))) if valid.any() else None
   except AssertionError as exc:
    valid=np.zeros((256,256),bool);bbox=raybox=basebox=None;qualified=False;err='Renderer index/stencil bounds assertion';delta=None
   rows.append(dict(core_id=core['core_id'],winding=w,core_xywh=[x,y,128,128],context_xywh=[x-64,y-64,256,256],supervised_pixels=int(core['supervised_pixels']),
    geometry_qualified=qualified,geometry_error=err,geometry_valid_pixels=int(valid.sum()),geometry_invalid_pixels=int((~valid).sum()),
    bbox_conservative_xyz=bbox,bbox_endpoint_ray_plus1p01_xyz=raybox,base_extrema_union_xyz=basebox,staged_vs_float64_max_absolute_difference=delta))
 need(len(rows)==265,'Fixed265 cores')
 pairs=[]
 for i,a in enumerate(rows):
  for b in rows[i+1:]:
   if a['winding']==b['winding'] or a['bbox_conservative_xyz'] is None or b['bbox_conservative_xyz'] is None:continue
   if intersects(a['bbox_conservative_xyz'],b['bbox_conservative_xyz']):pairs.append(dict(a=a['core_id'],b=b['core_id'],winding_a=a['winding'],winding_b=b['winding'],both_geometry_qualified=a['geometry_qualified'] and b['geometry_qualified']))
 folds=fold_rows(rows)
 for pin in before:need(sha(P/pin['relative'])==pin['sha256'],'Frozen input changed')
 result=dict(status='COMPLETE_GEOMETRY_ONLY',created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=plan['protocol_sha256'],physical_frame='native20250728140407 XYZ voxels,9.362um',context_pixels=256*256,depth_offsets=[j-13.5 for j in range(28)],conservative_margin_per_axis=14.51,cohort_counts={s['winding']:s['cores'] for s in site_reports},sites=site_reports,cores=rows,cross_winding_intersections=pairs,folds=folds,ink_labels_used_for_selection=False,CT_chunks_fetched=0,model_forwards=0,GPU_spend_GBP=0,limitation='Conservative boxes reject possible shared input support; intersections do not prove actual same tissue/voxels; invalid held-out geometry blocks full fold admission.')
 save(P/'RESULTS.json',result)
 for f in folds:
  (P/f"{f['held_out']}-retained-training-core-ids.txt").write_text(''.join(x+'\n' for x in f['retained_training_core_ids']))
 print(json.dumps(dict(cores=len(rows),qualified=sum(r['geometry_qualified'] for r in rows),cross_winding_pairs=len(pairs),folds=[{k:f[k] for k in ['held_out','evaluation_cores','retained_training_cores','retained_training_supervised_pixels','fold_geometry_admitted']} for f in folds]),indent=2))
if __name__=='__main__':main()
