"""Freeze geometry rectangles, sample sites and source whitelist before access."""
import sys,json,itertools,shutil,datetime
import numpy as np,tifffile,cv2
from common import *
sys.path.insert(0,str(R/'fortyeighth-experiment'));from support import reconstruct,corner_envelope
assert not (P/'RENDER_GO.json').exists()
priorpath=R/'fiftieth-experiment/final-verification.json'
assert pin(priorpath)['sha256']=='a0c640dbebe0ce48af00edb2aaaeeab2e21c16376afac65e43fd65fe2376a5ca'
prior=json.loads(priorpath.read_text());lookup={r['path']:r for r in prior['records']}
oldgo=R/'fortyninth-experiment/GO.json';assert pin(oldgo)==lookup[str(oldgo)]
for r in json.loads(oldgo.read_text())['inputs']:lookup.setdefault(r['path'],r)
old=R/'fortyninth-experiment';plan=json.loads((old/'TRANSFER_PLAN.json').read_text());masks=np.load(old/'work/target-planning-masks.npz');G=LAB/'data/experiments/pherc0800-larger-geometry/20251028222030'
paths=[G/f'mesh.{a}.tif' for a in 'xyz']+[G/'mesh.meta.json',old/'TRANSFER_PLAN.json',old/'OUTPUT_SUPPORT_PLAN.json',old/'work/target-planning-masks.npz',old/'work/planned-output-support.npz',R/'fortyeighth-experiment/support.py']
paths += [R/'fortysixth-experiment/work/squashfs-root'/s for s in ['AppRun','usr/bin/vc_render_tifxyz']]
for p in paths:assert pin(p)==lookup[str(p)],str(p)
mesh=np.stack([tifffile.imread(G/f'mesh.{a}.tif') for a in 'xyz'],-1)
b,n,bv,nv=reconstruct(mesh,json.loads((G/'mesh.meta.json').read_text())['scale'],(2580,2580));known=masks['required_context']&bv&nv
assert np.array_equal(nv,masks['valid_geometry']) and np.array_equal(known,masks['required_valid'])
np.save(W/'observed-mask.npy',known);np.savez_compressed(W/'render-geometry.npz',base=b,normal=n)
active={};rectangles=[]
for y in range(known.shape[0]+1):
 line=known[y] if y<known.shape[0] else np.zeros(known.shape[1],bool);d=np.diff(np.pad(line.astype(int),(1,1)));runs=set(zip(np.where(d==1)[0],np.where(d==-1)[0]))
 for run in list(active):
  if run not in runs:
   start=active.pop(run);rectangles.append([int(run[0]),start,int(run[1]-run[0]),y-start])
 for run in sorted(runs):active.setdefault(run,y)
rectangles.sort(key=lambda r:(r[1],r[0]));cover=np.zeros(known.shape,np.uint8);cases=[];samples=[]
keys=['0/'+'/'.join(map(str,k)) for k in plan['source_chunk_indices_zyx']]
for i,(x,y,w,h) in enumerate(rectangles):
 cover[y:y+h,x:x+w]+=1
 points=[[int(yy),int(xx)] for yy in np.unique(np.linspace(y,y+h-1,min(h,8)).round().astype(int)) for xx in np.unique(np.linspace(x,x+w-1,min(w,8)).round().astype(int))]
 cases.append(dict(name=f'rect-{i:03}',crop_xywh=[x,y,w,h],segmentation=str(W/'target.tifxyz'),source_keys=keys));samples.append(dict(name=f'rect-{i:03}',points_yx=points))
assert np.array_equal(cover,known.astype(np.uint8)) and len(cases)==104
# Fixed lattice; choose nearest known32x32 square using geometry only.
s=np.pad((~known).astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1);bad=s[32:,32:]-s[:-32,32:]-s[32:,:-32]+s[:-32,:-32];yy,xx=np.where(bad==0);sites=[]
for cy in [640,1280,1920]:
 for cx in [640,1280,1920]:
  order=np.lexsort((xx,yy,(xx+16-cx)**2+(yy+16-cy)**2));j=order[0];x,y=int(xx[j]),int(yy[j]);sites.append(dict(name=f'lattice-{cx}-{cy}',crop_xywh=[x,y,32,32]));samples.append(dict(name=sites[-1]['name'],points_yx=[[y+dy,x+dx] for dy in range(32) for dx in range(32)]))
# Recheck geometric source bounds against old frozen384-chunk envelope.
lo,hi=corner_envelope(b[known],n[known]);lo=lo.astype(np.int64)//128;hi=hi.astype(np.int64)//128;computed=set()
for delta in itertools.product([0,1],repeat=3):
 ix=lo+delta;keep=(ix<=hi).all(-1);computed.update('0/'+'/'.join(map(str,v[::-1])) for v in np.unique(ix[keep],axis=0))
assert computed==set(keys) and len(keys)==384
retained=[]
for key in keys:
 for previous in ['fiftieth-experiment','fortyninth-experiment']:
  f=R/previous/'work/source-cache'/key
  if f.exists():assert pin(f)==lookup[str(f)];retained.append(dict(key=key,**pin(f)));break
meshdir=W/'target.tifxyz';meshdir.mkdir()
for a in ['x.tif','y.tif','z.tif','meta.json']:shutil.copyfile(G/f'mesh.{a}',meshdir/a)
config=W/'vc3d-config';config.mkdir();(config/'VC3D.ini').write_text('[viewer]\nremote_cache_dir='+str(W/'renderer-cache')+'\n[perf]\nremote_cache_delta3d=false\n')
save(W/'sampling-sites.json',samples)
go=dict(status='FROZEN',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=pin(P/'PROTOCOL.md'),pins=[pin(p) for p in paths+[P/'prepare.py',P/'common.py',P/'render.py',P/'verify_render.py',W/'sampling-sites.json',W/'observed-mask.npy',W/'render-geometry.npz']],source_url=json.loads((old/'GO.json').read_text())['source_url'],source_keys=keys,retained_chunks=retained,new_chunks=len(keys)-len(retained),payload_bound_bytes=len(keys)*128**3,full_shape_yx=[2580,2580],scoring_xywh=[512,512,1536,1536],observed_pixels=int(known.sum()),cases=cases,distributed_sites=sites,grids=plan['rows'],depth=66,flip_normals=True,voxel_size_um=8.64)
save(P/'RENDER_GO.json',go);print(json.dumps({k:go[k] for k in ['status','new_chunks','payload_bound_bytes','observed_pixels','distributed_sites']}))
