"""Independent protocol-derived PHerc0800 geometry reconstruction; no reference imports."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import json,hashlib,sys,resource,time
import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.ndimage import minimum_filter
import tifffile
P=Path(__file__).parent/'packet'; OUT=Path(__file__).parent/'output-locked'; OUT.mkdir(exist_ok=True)
GO=json.loads((P/'frozen/GO.json').read_text())
def load(s):
 p=P/'inputs/meshes'/s
 return np.stack([tifffile.imread(p/f'mesh.{a}.tif') for a in 'xyz'],axis=-1),json.loads((p/'mesh.meta.json').read_text())['scale'],json.loads((p/'surface.level0.zarray.json').read_text())['shape'][1:]
def interpolate(m,gx,gy):
 ix=np.floor(gx).astype(int); iy=np.floor(gy).astype(int)
 valid=(ix>=0)&(iy>=0)&(ix+1<m.shape[1])&(iy+1<m.shape[0])
 ix=np.clip(ix,0,m.shape[1]-2); iy=np.clip(iy,0,m.shape[0]-2)
 a,b,c,d=m[iy,ix],m[iy,ix+1],m[iy+1,ix],m[iy+1,ix+1]
 valid &= np.all(np.isfinite(a)&(a>0)&np.isfinite(b)&(b>0)&np.isfinite(c)&(c>0)&np.isfinite(d)&(d>0),axis=-1)
 tx=(gx-ix).astype(gx.dtype)[...,None];ty=(gy-iy).astype(gy.dtype)[...,None]
 base=(a*(1-tx)*(1-ty)+b*tx*(1-ty)+c*(1-tx)*ty+d*tx*ty).astype(np.float32)
 return base,valid

def historical(m,s,shape):
 yy,xx=np.indices(shape,dtype=np.float64)
 return interpolate(m,(xx+.5)*s[0],(yy+.5)*s[1])
def tile(m,s,x0,y0,w,h):
 yy,xx=np.indices((h,w),dtype=np.float32)
 # Tile-local arithmetic plus float32 tile origin; four mesh-cell halo.
 hx=max(0,int(np.floor(x0*s[0]))-4);hy=max(0,int(np.floor(y0*s[1]))-4)
 gx=(xx+np.float32(.5))*np.float32(s[0])+np.float32(x0*s[0]-hx)
 gy=(yy+np.float32(.5))*np.float32(s[1])+np.float32(y0*s[1]-hy)
 sub=m[hy:,hx:]
 base,valid=interpolate(sub,gx,gy)
 nx=np.floor(gx+.5).astype(int)+hx;ny=np.floor(gy+.5).astype(int)+hy
 nv=(nx>0)&(ny>0)&(nx<m.shape[1]-1)&(ny<m.shape[0]-1)
 nx=np.clip(nx,1,m.shape[1]-2);ny=np.clip(ny,1,m.shape[0]-2)
 c,l,r,u,d=m[ny,nx],m[ny,nx-1],m[ny,nx+1],m[ny-1,nx],m[ny+1,nx]
 for a in [c,l,r,u,d]:nv &= np.all(np.isfinite(a)&(a>0),axis=-1)
 n=-np.cross(r-l,d-u); length=np.sqrt(np.sum(n*n,axis=-1,dtype=np.float32));nv &= np.isfinite(length)&(length>1e-12)
 n=n/np.where(length>0,length,1)[...,None]
 return base,n,valid,nv

def largest(mask):
 # Binary search square size using integral sums, deterministic smallest y/x.
 h,w=mask.shape; a=np.pad(mask.astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1)
 def test(k): return (a[k:,k:]-a[:-k,k:]-a[k:,:-k]+a[:-k,:-k])==k*k
 lo,hi=0,min(h,w)
 while lo<hi:
  mid=(lo+hi+1)//2
  if test(mid).any():lo=mid
  else:hi=mid-1
 if not lo:return {'side':0,'xywh':None}
 y,x=np.argwhere(test(lo))[0];return {'side':lo,'xywh':[int(x),int(y),lo,lo]}
def starts(n):return sorted(set(list(range(0,n-127,64))+[n-128]))
def exact_context(mask):
 h,w=mask.shape; result=np.ones_like(mask); patches=[]
 for y in starts(h):
  for x in starts(w):
   ok=bool(mask[y:y+128,x:x+128].all());patches.append([x,y,ok])
   if not ok:result[y:y+128,x:x+128]=False
 return result,patches

def main():
 t=time.time(); checks={}
 for file in ['INPUT_MANIFEST.json','PACKET_MANIFEST.json']:
  d=json.loads((P/file).read_text())
  for r in d['files']:
   b=(P/r['path']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
  checks[file]=len(d['files'])
 old=[];old_counts={}
 for s,(x,y,w,h) in GO['previous_surface_extents']:
  m,sc,shape=load(s);b,v=historical(m,sc,shape);v=v[y:y+h,x:x+w];b=b[y:y+h,x:x+w];old.append(b[v].astype(np.float64));old_counts[s]=int(v.sum())
 centers=np.concatenate(old);del old;assert len(centers)==3260832
 tree=cKDTree(centers); print('Old centers',old_counts,flush=True)
 receipts=[]
 for file in sorted((P/'inputs/archived-point-checks').glob('*.json')):
  d=json.loads(file.read_text());s=d['segment_id'];m,sc,shape=load(s)
  for pt in d['points']:
   x,y=pt.get('surface_xy',pt.get('source_xy'));x0=x//128*128;y0=y//128*128
   b,n,v,nv=tile(m,sc,x0,y0,min(128,shape[1]-x0),min(128,shape[0]-y0));i=(y-y0,x-x0)
   receipts.append({'segment':s,'xy':[x,y],'base_max_abs':float(np.max(np.abs(b[i]-pt['base_ct_xyz']))),'normal_max_abs':float(np.max(np.abs(n[i]-pt['normal_xyz_flipped']))),'valid':bool(v[i]&nv[i])})
 print('Receipts',max(r['base_max_abs'] for r in receipts),max(r['normal_max_abs'] for r in receipts),flush=True)
 results=[];components=[]
 for s in GO['candidate_segments']:
  m,sc,shape=load(s);h,w=shape;historical_b,historical_v=historical(m,sc,shape)
  masks={k:np.zeros(shape,bool) for k in ['bilinear_valid','normal_valid','inside_native','far_surface','old_ct_clear','point_support']}; maxerr=0.
  for y in range(0,h,128):
   for x in range(0,w,128):
    b,n,bv,nv=tile(m,sc,x,y,min(128,w-x),min(128,h-y));hh,ww=bv.shape;sl=np.s_[y:y+hh,x:x+ww]
    if bv.any():maxerr=max(maxerr,float(np.max(np.abs(b[bv]-historical_b[sl][bv]))))
    lo=np.floor(b.astype(float)-32.5*np.abs(n.astype(float))-.01).astype(np.int32)
    hi=np.floor(b.astype(float)+32.5*np.abs(n.astype(float))+.01).astype(np.int32)+1
    inside=(lo>=0).all(-1)&(hi<np.array(GO['target_shape_zyx'][::-1])).all(-1)
    clear=np.ones(bv.shape,bool)
    for box in GO['previous_CT_boxes']:clear &= ~((hi>=box['lo_xyz']).all(-1)&(lo<=box['hi_xyz_inclusive']).all(-1))
    far=np.zeros(bv.shape,bool);dist,_=tree.query(b[bv].astype(float),workers=4,distance_upper_bound=np.nextafter(51.1,np.inf));far[bv]=dist>51.1
    for k,v in [('bilinear_valid',bv),('normal_valid',nv),('inside_native',inside),('far_surface',far),('old_ct_clear',clear),('point_support',bv&nv&inside&far&clear)]:masks[k][sl]=v
  point=masks['point_support'];allorigin=minimum_filter(point,size=257,mode='constant',cval=0);exact,patches=exact_context(point);assert not (allorigin&~exact).any()
  masks.update(all_origin_context=allorigin,fixed_grid_context=exact)
  labels=np.load(P/'inputs/phase47-components'/f'{s}.npz',allow_pickle=False)['far_component_labels'];full=labels[np.minimum((np.arange(h)+10)//20,labels.shape[0]-1)[:,None],np.minimum((np.arange(w)+10)//20,labels.shape[1]-1)[None,:]]
  for lab in sorted(np.unique(labels)):
   if lab==0:continue
   cm=full==lab;a=allorigin&cm;e=exact&cm; sq=largest(e)
   item={'segment':s,'component':int(lab),'sampled_size':int((labels==lab).sum()),'point_area':int((point&cm).sum()),'all_origin_area':int(a.sum()),'all_origin_square':largest(a),'fixed_grid_area':int(e.sum()),'fixed_grid_square':sq}
   if sq['side']>=128:
    xx,yy,sw,sh=sq['xywh'];item['contributing_patches_xywh']=[[px,py,128,128] for px,py,ok in patches if px<xx+sw and px+128>xx and py<yy+sh and py+128>yy];assert all(ok for px,py,ok in patches if px<xx+sw and px+128>xx and py<yy+sh and py+128>yy)
   components.append(item)
  bv=masks['bilinear_valid'];nv=bv&masks['normal_valid'];native=nv&masks['inside_native'];far=native&masks['far_surface'];clear=far&masks['old_ct_clear']
  row={'segment':s,'shape_yx':shape,'historical_valid':int(historical_v.sum()),'base_max_abs_error':maxerr,'counts':{k:int(v.sum()) for k,v in masks.items()},'sequential_counts':{'bilinear':int(bv.sum()),'normal':int(nv.sum()),'native':int(native.sum()),'far_surface':int(far.sum()),'old_ct_clear':int(clear.sum()),'all_origin':int(allorigin.sum()),'fixed_grid':int(exact.sum())}}
  results.append(row);np.savez_compressed(OUT/f'{s}-masks.npz',**masks);print(json.dumps(row),flush=True)
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<7*1024*1024
  print('resource_peak_rss_kib',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
 preferred_size=max(c['sampled_size'] for c in components if c['segment']=='20251028225813')
 components.sort(key=lambda r:(0 if r['segment']=='20251028225813' and r['sampled_size']==preferred_size else 1,-r['sampled_size'],r['segment'],r['component']))
 data={'source_zip_sha256':'8d73a97c6fba42436d1bdb0397cbb00ba59822e437bf2a6b4998d66fafe523ea','manifest_checks':checks,'old_counts':old_counts,'receipts':receipts,'meshes':results,'components_ranked':components,'all_origin_selected':next((c for c in components if c['all_origin_square']['side']>=128),None),'fixed_grid_selected':next((c for c in components if c['fixed_grid_square']['side']>=128),None),'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'tifffile':tifffile.__version__,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-t,'kdtree_workers':4,'blas_threads':1}}
 (OUT/'independent-results.json').write_text(json.dumps(data,indent=2));print('DONE',data['environment'],flush=True)
if __name__=='__main__':main()
