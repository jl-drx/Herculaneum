"""POST-REFERENCE numerical-convention reconciliation, not independent evidence.
Implements reviewed archived float32 axis/interpolation/normal operations explicitly.
Does not import or run packet code. Compares to frozen independent masks.
"""
from audit_locked import *
DEST=Path(__file__).parent/'reconciliation';DEST.mkdir(exist_ok=True)
def archived_geometry(m,sc,shape):
 h,w=shape;s=np.asarray(sc,np.float32);center=np.asarray([m.shape[1]/2/float(s[0]),m.shape[0]/2/float(s[1])],np.float32)
 def axis(n,k):
  pixels=np.arange(n);tile=pixels//128*128
  off=np.asarray(-.5*(n-1)+tile,np.float32)
  upper=(off+center[k]).astype(np.float32)*s[k]
  origin=np.asarray(upper.astype(float)-4*float(s[k]),np.float32)
  return (origin+(pixels-tile+4).astype(np.float32)*s[k]).astype(np.float32)
 gx,gy=axis(w,0),axis(h,1);base=np.zeros((h,w,3),np.float32);normal=np.zeros_like(base);valid=np.zeros((h,w),bool);bv=np.zeros_like(valid)
 good=np.isfinite(m).all(-1)&(m>0).all(-1)
 for y in range(0,h,128):
  X,Y=np.meshgrid(gx,gy[y:y+128]);ix=np.floor(X).astype(int);iy=np.floor(Y).astype(int);ok=(ix>=0)&(iy>=0)&(ix+1<m.shape[1])&(iy+1<m.shape[0]);ix=np.clip(ix,0,m.shape[1]-2);iy=np.clip(iy,0,m.shape[0]-2)
  ok &= good[iy,ix]&good[iy,ix+1]&good[iy+1,ix]&good[iy+1,ix+1]
  wx=(X-ix).astype(np.float32)[...,None];wy=(Y-iy).astype(np.float32)[...,None]
  b=(m[iy,ix]*(1-wx)+m[iy,ix+1]*wx)*(1-wy)+(m[iy+1,ix]*(1-wx)+m[iy+1,ix+1]*wx)*wy
  nx=np.floor(X.astype(float)+.5).astype(int);ny=np.floor(Y.astype(float)+.5).astype(int);nv=ok&(nx>=1)&(ny>=1)&(nx+1<m.shape[1])&(ny+1<m.shape[0]);nx=np.clip(nx,1,m.shape[1]-2);ny=np.clip(ny,1,m.shape[0]-2)
  nv &= good[ny,nx]&good[ny,nx-1]&good[ny,nx+1]&good[ny-1,nx]&good[ny+1,nx]
  n=np.cross(m[ny,nx+1]-m[ny,nx-1],m[ny+1,nx]-m[ny-1,nx]);length=np.sqrt(n[...,0]**2+n[...,1]**2+n[...,2]**2);nv &= np.isfinite(length)&(length>0)
  with np.errstate(invalid='ignore',divide='ignore'):
   n=-n*(np.float32(1)/length)[...,None];length=np.sqrt(n[...,0]**2+n[...,1]**2+n[...,2]**2);n=n/length[...,None]
  sl=np.s_[y:y+len(Y)];base[sl]=np.where(ok[...,None],b,0);normal[sl]=np.where(nv[...,None],n,0);bv[sl]=ok;valid[sl]=nv
 return base,normal,bv,valid

def main():
 old=[]
 for seg,(x,y,w,h) in GO['previous_surface_extents']:
  m,sc,shape=load(seg);b,v=historical(m,sc,shape);old.append(b[y:y+h,x:x+w][v[y:y+h,x:x+w]].astype(float))
 tree=cKDTree(np.concatenate(old));del old;rows=[];fixtures=[]
 expected={r['segment']:r for r in json.loads((P/'expected/RESULTS.json').read_text())['meshes']}
 for seg in GO['candidate_segments']:
  m,sc,shape=load(seg);b,n,bv,nv=archived_geometry(m,sc,shape);lo=np.floor(b.astype(float)-32.5*n.astype(float)-.01);hi=np.floor(b.astype(float)+32.5*n.astype(float)+.01)+1
  # Sign-independent endpoint extrema, each normal multiply is float32 as in reviewed code.
  e1=b.astype(float)-np.float32(32.5)*n;e2=b.astype(float)+np.float32(32.5)*n;lo=np.floor(np.minimum(e1,e2)-.01);hi=np.floor(np.maximum(e1,e2)+.01)+1
  inside=nv&(lo>=0).all(-1)&(hi<np.array(GO['target_shape_zyx'][::-1])).all(-1);clear=np.ones(shape,bool)
  for box in GO['previous_CT_boxes']:clear &= ~((hi>=box['lo_xyz']).all(-1)&(lo<=box['hi_xyz_inclusive']).all(-1))
  dist=np.full(shape,np.inf)
  for y in range(0,shape[0],256):
   v=bv[y:y+256];q=b[y:y+256][v].astype(float);temp=dist[y:y+256];temp[v]=tree.query(q,workers=4)[0]
  far=dist>51.1;point=inside&clear&far;allorigin=minimum_filter(point,size=257,mode='constant',cval=0);fixed,patches=exact_context(point)
  oldmask=np.load(Path(__file__).parent/'output'/f'{seg}-masks.npz');diff=np.argwhere(point!=oldmask['point_support']);row={'segment':seg,'normal_valid':int(nv.sum()),'inside_CT':int(inside.sum()),'outside_old_surface_bound':int((inside&far).sum()),'old_CT_intersections_on_geometry_valid':int((inside&~clear).sum()),'point_support':int(point.sum()),'context_core':int(allorigin.sum()),'point_mask_different_pixels_vs_independent':len(diff),'allorigin_mask_different_pixels_vs_independent':int((allorigin!=oldmask['all_origin_context']).sum()),'fixed_grid_mask_different_pixels_vs_independent':int((fixed!=oldmask['fixed_grid_context']).sum())}
  row['retained_expected_deltas']={k:row[k]-expected[seg][k] for k in ['normal_valid','inside_CT','outside_old_surface_bound','old_CT_intersections_on_geometry_valid','point_support','context_core']}
  for y,x in diff:
   xx=x//128*128;yy=y//128*128;ib,inn,_,_=tile(m,sc,xx,yy,min(128,shape[1]-xx),min(128,shape[0]-yy));bi=ib[y-yy,x-xx];di=float(tree.query(bi.astype(float))[0]);fixtures.append({'segment':seg,'xy':[int(x),int(y)],'independent_base_xyz':bi.tolist(),'archived_base_xyz':b[y,x].tolist(),'independent_nearest_distance':di,'archived_nearest_distance':float(dist[y,x]),'independent_passes':bool(oldmask['point_support'][y,x]),'archived_passes':bool(point[y,x])})
  if seg=='20251028225813':
   patch_union=point[192:512,960:1280];row['preferred_candidate_all_102400_source_pixels_supported']=bool(patch_union.all());row['preferred_candidate_nearest_old_distance_min']=float(dist[192:512,960:1280].min())
   def margins(l,h):
    boundary=float(np.minimum(l,np.array(GO['target_shape_zyx'][::-1])-1-h).min())
    boxes=[]
    for box in GO['previous_CT_boxes']:
     separation=np.maximum(l-np.array(box['hi_xyz_inclusive']),np.array(box['lo_xyz'])-h).max(-1)
     boxes.append(float(separation.min()))
    return {'native_boundary_min_integer_index_gap':boundary,'previous_box_min_separating_axis_gaps_in_GO_order':boxes,'previous_boxes_min_gap':min(boxes)}
   row['preferred_candidate_archived_margins']=margins(lo[192:512,960:1280],hi[192:512,960:1280])
   ib=np.zeros((320,320,3),np.float32);inn=np.zeros_like(ib)
   for yy in range(128,512,128):
    for xx in range(896,1280,128):
     tb,tn,_,_=tile(m,sc,xx,yy,128,128);ya,yb=max(192,yy),min(512,yy+128);xa,xb=max(960,xx),min(1280,xx+128)
     ib[ya-192:yb-192,xa-960:xb-960]=tb[ya-yy:yb-yy,xa-xx:xb-xx];inn[ya-192:yb-192,xa-960:xb-960]=tn[ya-yy:yb-yy,xa-xx:xb-xx]
   ild=np.floor(ib.astype(float)-32.5*np.abs(inn.astype(float))-.01);ihd=np.floor(ib.astype(float)+32.5*np.abs(inn.astype(float))+.01)+1
   row['preferred_candidate_independent_margins']=margins(ild,ihd)
   row['preferred_candidate_independent_nearest_old_distance_min']=float(tree.query(ib.reshape(-1,3).astype(float),workers=4)[0].min())
   row['preferred_candidate_all_16_patches_supported_both']=bool(patch_union.all() and oldmask['point_support'][192:512,960:1280].all())
  np.savez_compressed(DEST/f'{seg}-masks.npz',point_support=point,all_origin_context=allorigin,fixed_grid_context=fixed);rows.append(row);print(json.dumps(row),flush=True)
 (DEST/'metrics.json').write_text(json.dumps({'rows':rows,'discrepancy_fixtures':fixtures,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
if __name__=='__main__':main()
