"""Prepare global partial canvases and freeze qualified batch cohorts."""
import sys,json,subprocess
import numpy as np,zarr,cv2
from common import *
from worker import cohorts
assert json.loads((P/'RENDER_VERIFICATION.json').read_text())['status']=='PASS'
V=W/'model-inputs';V.mkdir();render=np.load(W/'render66.npy',mmap_mode='r');known=np.load(W/'observed-mask.npy');S=R/'fortyfifth-experiment/source-evidence/pherc343/ink/scripts'
for phase,py,px in [('plain',0,0),('padded',40,200)]:
 shape=(2580+py,2580+px);group=zarr.open_group(str(V/(phase+'.zarr')),mode='w',zarr_format=2);a=group.create_array('0',shape=(66,*shape),chunks=(66,128,128),dtype='u1')
 # Only write intersecting observed blocks, never claim other zeros as CT.
 for y in range(0,2580,128):
  for x in range(0,2580,128):
   h,w=min(128,2580-y),min(128,2580-x)
   if known[y:y+h,x:x+w].any():a[:,y+py:y+py+h,x+px:x+px+w]=render[:,y:y+h,x:x+w]
 group.attrs.update(dict(scope='Partial observed context only; unknown outside known-mask. Padding is known synthetic zero.',native_spacing_um=8.64))
 km=np.pad(known,((py,0),(px,0)),constant_values=True);np.save(V/(phase+'-known.npy'),km)
 subprocess.run([sys.executable,'-B',str(S/'hecate_prep.py'),'--src',str(V/(phase+'.zarr')),'--out',str(V/('hecate-'+phase+'.zarr')),'--native-um','8.64','--native-um-z','8.64','--target-um','9.6','--z-center','32.5','--nz','16'],check=True)
 hh,ww=round(shape[0]*.9),round(shape[1]*.9);hk=cv2.resize(km.astype(np.float32),(ww,hh),interpolation=cv2.INTER_AREA)>=1;np.save(V/('hecate-'+phase+'-known.npy'),hk)
rv=json.loads((P/'RENDER_VERIFICATION.json').read_text());grids=rv['grids']
for g in grids:g['cohorts_yx']=cohorts(g['family'],g['phase'],g)
old=json.loads((R/'fortyninth-experiment/INFERENCE_QUALIFICATION.json').read_text());models=[]
for r in old['models']:assert pin(r['checkpoint']['path'])==r['checkpoint'];models.append(dict(name=r['model'],checkpoint=r['checkpoint']))
paths=[P/f for f in ['PROTOCOL.md','common.py','worker.py','prepare_inputs.py','qualify_batches.py','freeze_inference.py','RENDER_GO.json','RENDER_VERIFICATION.json']]+[W/'render66.npy',W/'observed-mask.npy',W/'output-support.npz',S/'hecate_prep.py',R/'fortyfifth-experiment/source-evidence/hecate-hecate.py']
paths += [p for p in V.rglob('*') if p.is_file()]
prior=json.loads((R/'fiftieth-experiment/final-verification.json').read_text());runtime=[]
for r in prior['records']:
 if r['path'].startswith(str(R/'fortysixth-experiment/runtime-source')+'/') and Path(r['path']).suffix in ['.py','.json']:assert pin(r['path'])==r;runtime.append(r)
go=dict(status='FROZEN_FOR_QUALIFICATION',models=models,pins=[pin(p) for p in paths],runtime_pins=runtime,core_xywh=[512,512,1536,1536],grids=grids,batch_size=4,threads=4,precision='fp32 CPU',gpu_spend_GBP=0)
save(P/'INPUT_GO.json',go);print(json.dumps(dict(status='FROZEN_FOR_QUALIFICATION',grids=[dict(family=g['family'],phase=g['phase'],patches=g['patch_count'],forward_batches=len(g['cohorts_yx']),nonempty_tiles=sum(map(len,g['cohorts_yx']))) for g in grids])))
