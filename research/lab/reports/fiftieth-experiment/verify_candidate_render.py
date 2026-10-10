"""Verify explicit physical metadata and actual candidate sampling; no inference."""
from pathlib import Path
import json,itertools,hashlib,sys
import numpy as np,zarr,cv2
from admit_candidate import pin,save
P=Path(__file__).resolve().parent;W=P/'work';R=P.parent
plan=json.loads((P/'RENDER_GO.json').read_text());assert json.loads((P/'RENDER_EXECUTION.json').read_text())['status']=='PASS'
oldgo=json.loads((R/'fortyninth-experiment/GO.json').read_text())
for rec in oldgo['inputs']:
 if rec['path'].endswith('vc_render_tifxyz') or rec['path'].endswith('AppRun'):assert pin(rec['path'])==rec
metadata=[]
for case in plan['cases']:
 group=zarr.open_group(str(W/(case['name']+'.zarr')),mode='r');attrs=dict(group.attrs)
 assert attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']==[8.64,8.64,8.64],attrs
 assert all(a.get('unit')=='micrometer' for a in attrs['multiscales'][0]['axes']),attrs
 metadata.append(dict(case=case['name'],attrs=attrs))
control=np.asarray(zarr.open_group(str(W/'metadata-control.zarr'),mode='r')['0']);old=np.load(R/'fortyninth-experiment/work/render66.npy');assert np.array_equal(control,old)
x0,y0,w,h=plan['candidate_render_xywh'];observed=np.load(W/'render-observed-mask.npy');g=np.load(W/'render-geometry.npz');b,n=g['base'][observed],g['normal'][observed]
assert g['valid'][observed].all()
a=np.zeros((66,h,w),np.uint8);placed=np.zeros((h,w),np.uint8)
for case in plan['cases'][1:]:
 x,y,cw,ch=case['crop_xywh'];sl=np.s_[y-y0:y-y0+ch,x-x0:x-x0+cw];raw=np.asarray(zarr.open_group(str(W/(case['name']+'.zarr')),mode='r')['0']);assert raw.shape==(66,ch,cw);a[:,sl[0],sl[1]]=raw;placed[sl]+=1
assert np.array_equal(placed,observed.astype(np.uint8))
chunks={tuple(map(int,key.split('/')[1:])):np.frombuffer((W/'source-cache'/key).read_bytes(),np.uint8).reshape(128,128,128) for key in plan['candidate_source_keys']}
def fma(a,b,c):return (a.astype(np.float64)*b+c).astype(np.float32)
def at(zyx):
 idx=zyx//128;out=np.empty(len(zyx),np.float32)
 for key in np.unique(idx,axis=0):
  m=np.all(idx==key,axis=1);local=zyx[m]%128;out[m]=chunks[tuple(key)][local[:,0],local[:,1],local[:,2]]
 return out
pred=np.zeros_like(a)
for k in range(66):
 xyz=fma(n,k-32.5,b);base=np.floor(xyz).astype(int);frac=xyz-base
 v={d:at((base+np.array(d))[:,::-1]) for d in itertools.product([0,1],repeat=3)}
 ix={(y,z):fma(frac[:,0],v[(1,y,z)]-v[(0,y,z)],v[(0,y,z)]) for y,z in itertools.product([0,1],repeat=2)}
 iy={z:fma(frac[:,1],ix[(1,z)]-ix[(0,z)],ix[(0,z)]) for z in [0,1]}
 pred[k,observed]=np.clip(fma(frac[:,2],iy[1]-iy[0],iy[0]),0,255).astype(np.uint8)
np.save(W/'candidate-render66.npy',a);np.save(W/'candidate-independent66.npy',pred)
diff=np.abs(a[:,observed].astype(np.int16)-pred[:,observed].astype(np.int16))
stats=dict(samples=int(diff.size),exact=int((diff==0).sum()),exact_fraction=float((diff==0).mean()),mean_abs=float(diff.mean()),max_abs=int(diff.max()),within_one_fraction=float((diff<=1).mean()))
# Independently prove the unknown mosaic strip is outside every scheduled input.
known=np.zeros((2300,2260),bool);known[y0:y0+h,x0:x0+w]=observed;support=[]
for grid in plan['grids']:
 py,px=(0,0) if grid['phase']=='plain' else (40,200);km=np.pad(known,((py,0),(px,0)))
 if grid['family']=='hecate':hh,ww=grid['grid_shape_yx'];km=cv2.resize(km.astype(np.float32),(ww,hh),interpolation=cv2.INTER_AREA)>=1.0
 for x,y,pw,ph in grid['patches_xywh']:assert km[y:y+ph,x:x+pw].all(),(grid['phase'],grid['family'],x,y)
 support.append(dict(family=grid['family'],phase=grid['phase'],all_input_pixels_observed=True,patches=grid['patch_count']))
gates=dict(metadata_control_byte_exact=True,correct_physical_metadata=True,all_model_context_observed=True,independent_candidate=stats['max_abs']<=1 and stats['mean_abs']<=.1 and stats['exact_fraction']>=.95)
result=dict(status='PASS' if all(gates.values()) else 'FAIL',gates=gates,candidate_comparison=stats,observed_mosaic_pixels=int(observed.sum()),unknown_mosaic_pixels=int((~observed).sum()),context_checks=support,metadata=metadata,script=pin(__file__),full_target_maps=0,new_model_inference=False)
save(P/'RENDER_VERIFICATION.json',result);print(json.dumps({k:v for k,v in result.items() if k!='metadata'}));sys.exit(0 if result['status']=='PASS' else 2)
