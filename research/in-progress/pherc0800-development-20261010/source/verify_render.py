"""Validate every rectangle, frozen distributed samples and complete context masks."""
import json,itertools,sys
import numpy as np,zarr,cv2
from common import *
go=json.loads((P/'RENDER_GO.json').read_text());assert json.loads((P/'RENDER_EXECUTION.json').read_text())['status']=='PASS'
known=np.load(W/'observed-mask.npy');volume=np.lib.format.open_memmap(W/'render66.npy',mode='w+',dtype='uint8',shape=(66,2580,2580));placed=np.zeros(known.shape,np.uint8)
for case in go['cases']:
 x,y,w,h=case['crop_xywh'];group=zarr.open_group(str(W/(case['name']+'.zarr')),mode='r');attrs=dict(group.attrs)
 assert attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']==[8.64]*3
 assert all(a.get('unit')=='micrometer' for a in attrs['multiscales'][0]['axes'])
 a=np.asarray(group['0']);assert a.shape==(66,h,w);assert known[y:y+h,x:x+w].all();volume[:,y:y+h,x:x+w]=a;placed[y:y+h,x:x+w]+=1
assert np.array_equal(placed,known.astype(np.uint8));volume.flush()
g=np.load(W/'render-geometry.npz');base,normal=g['base'],g['normal'];sites=json.loads((W/'sampling-sites.json').read_text());points=np.asarray([pt for site in sites for pt in site['points_yx']],np.int32)
b,n=base[points[:,0],points[:,1]],normal[points[:,0],points[:,1]]
chunks={key:np.memmap(W/'source-cache'/key,mode='r',dtype=np.uint8,shape=(128,128,128)) for key in go['source_keys']}
def fma(a,b,c):return (a.astype(np.float64)*b+c).astype(np.float32)
def at(zyx):
 ix=zyx//128;out=np.empty(len(zyx),np.float32)
 for key in np.unique(ix,axis=0):
  m=np.all(ix==key,axis=1);loc=zyx[m]%128;k='0/'+'/'.join(map(str,key));out[m]=chunks[k][loc[:,0],loc[:,1],loc[:,2]]
 return out
pred=np.empty((66,len(points)),np.uint8)
for k in range(66):
 xyz=fma(n,k-32.5,b);bi=np.floor(xyz).astype(int);frac=xyz-bi
 vals={d:at((bi+np.array(d))[:,::-1]) for d in itertools.product([0,1],repeat=3)}
 ix={(y,z):fma(frac[:,0],vals[(1,y,z)]-vals[(0,y,z)],vals[(0,y,z)]) for y,z in itertools.product([0,1],repeat=2)}
 iy={z:fma(frac[:,1],ix[(1,z)]-ix[(0,z)],ix[(0,z)]) for z in [0,1]}
 pred[k]=np.clip(fma(frac[:,2],iy[1]-iy[0],iy[0]),0,255).astype(np.uint8)
actual=volume[:,points[:,0],points[:,1]];diff=np.abs(actual.astype(np.int16)-pred.astype(np.int16))
def stats(d):return dict(samples=int(d.size),exact_fraction=float((d==0).mean()),mean_abs=float(d.mean()),max_abs=int(d.max()))
rows=[];offset=0
for site in sites:
 count=len(site['points_yx']);s=stats(diff[:,offset:offset+count]);offset+=count;s['name']=site['name'];s['pass']=s['exact_fraction']>=.95 and s['mean_abs']<=.1 and s['max_abs']<=1;rows.append(s)
np.savez_compressed(W/'render-verification-samples.npz',actual=actual,independent=pred,points_yx=points)
# Recompute full global contributor support using the actually observed mask.
common=np.ones((1536,1536),bool);out={};grids=[]
for row in go['grids']:
 py,px=(0,0) if row['phase']=='plain' else (40,200);padded=np.pad(known,((py,0),(px,0)),constant_values=True);h,w=padded.shape;hh,ww=row['grid_yx'];patch=row['patch']
 km=padded if row['family']=='villa' else cv2.resize(padded.astype(np.float32),(ww,hh),interpolation=cv2.INTER_AREA)>=1
 integral=np.pad((~km).astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1);bad=np.zeros((hh,ww),bool);coverage=np.zeros_like(bad);selected=[]
 for y,x in row['patches_yx']:
  coverage[y:y+patch,x:x+patch]=True
  if integral[y+patch,x+patch]-integral[y,x+patch]-integral[y+patch,x]+integral[y,x]:bad[y:y+patch,x:x+patch]=True
  else:selected.append([x,y,patch,patch])
 if row['family']=='hecate':bad=cv2.resize(bad.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>0;coverage=cv2.resize(coverage.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>=1
 support=(~bad&coverage)[512+py:2048+py,512+px:2048+px];out[row['family']+'_'+row['phase']]=support;common&=support
 grids.append(dict(family=row['family'],phase=row['phase'],grid_shape_yx=row['grid_yx'],patches_xywh=selected,patch_count=len(selected),all_contributing_grid_patches=len(row['patches_yx'])))
old=np.load(R/'fortyninth-experiment/work/planned-output-support.npz');assert np.array_equal(common,old['common'])
for k in out:assert np.array_equal(out[k],old[k]),k
np.savez_compressed(W/'output-support.npz',common=common,**out)
result=dict(status='PASS' if all(r['pass'] for r in rows) else 'FAIL',all_rectangles_correct_metadata=True,observed_pixels=int(known.sum()),sampling_checks=rows,pooled=stats(diff),common_support_pixels=int(common.sum()),grids=grids,script=pin(__file__),render=pin(W/'render66.npy'),samples=pin(W/'render-verification-samples.npz'))
save(P/'RENDER_VERIFICATION.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['sampling_checks','grids']}));assert result['status']=='PASS'
