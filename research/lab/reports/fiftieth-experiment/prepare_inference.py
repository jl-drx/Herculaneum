"""Materialize globally aligned, explicitly partial canvases and freeze model work."""
from pathlib import Path
import sys,hashlib,json,subprocess
import numpy as np,zarr,cv2
from admit_candidate import pin,save
P=Path(__file__).resolve().parent;W=P/'work';R=P.parent;V=W/'model-inputs';V.mkdir()
g=json.loads((P/'RENDER_GO.json').read_text());assert json.loads((P/'RENDER_VERIFICATION.json').read_text())['status']=='PASS'
render=np.load(W/'candidate-render66.npy');known=np.load(W/'render-observed-mask.npy');x0,y0,w,h=g['candidate_render_xywh']
S=R/'fortyfifth-experiment/source-evidence/pherc343/ink/scripts'
for phase,py,px in [('plain',0,0),('padded',40,200)]:
 shape=(2300+py,2260+px);v=zarr.open_group(str(V/(phase+'.zarr')),mode='w',zarr_format=2);a=v.create_array('0',shape=(66,*shape),chunks=(66,128,128),dtype='u1');a[:,y0+py:y0+py+h,x0+px:x0+px+w]=render
 v.attrs.update(dict(scope='Only pinned observed crop mask is CT; other zeros are unknown or intentional recipe padding',native_spacing_um=8.64,crop_xywh=[x0+px,y0+py,w,h]))
 km=np.zeros(shape,bool);km[y0+py:y0+py+h,x0+px:x0+px+w]=known;np.save(V/(phase+'-known.npy'),km)
 subprocess.run([sys.executable,'-B',str(S/'hecate_prep.py'),'--src',str(V/(phase+'.zarr')),'--out',str(V/('hecate-'+phase+'.zarr')),'--native-um','8.64','--native-um-z','8.64','--target-um','9.6','--z-center','32.5','--nz','16'],check=True)
 hh,ww=round(shape[0]*.9),round(shape[1]*.9);hk=cv2.resize(km.astype(np.float32),(ww,hh),interpolation=cv2.INTER_AREA)>=1.0;np.save(V/('hecate-'+phase+'-known.npy'),hk)
 for grid in [r for r in g['grids'] if r['phase']==phase]:
  m=hk if grid['family']=='hecate' else km
  for x,y,pw,ph in grid['patches_xywh']:assert m[y:y+ph,x:x+pw].all()
old=json.loads((R/'fortyninth-experiment/INFERENCE_QUALIFICATION.json').read_text());models=[]
for row in old['models']:
 ck=pin(row['checkpoint']['path']);assert ck==row['checkpoint'];models.append(dict(name=row['model'],checkpoint=ck))
paths=[P/'INFERENCE_PROTOCOL.md',P/'run_inference.py',P/'prepare_inference.py',P/'RENDER_GO.json',P/'RENDER_VERIFICATION.json',W/'candidate-render66.npy',W/'render-observed-mask.npy',S/'hecate_prep.py',R/'fortyfifth-experiment/source-evidence/hecate-hecate.py']
paths += [p for p in V.rglob('*') if p.is_file()]
prior=json.loads((R/'fortyninth-experiment/final-verification.json').read_text());runtime=[]
for r in prior['records']:
 if r['path'].startswith(str(R/'fortysixth-experiment/runtime-source')+'/') and Path(r['path']).suffix in ['.py','.json']:
  assert pin(r['path'])==r;runtime.append(r)
assert runtime
save(P/'INFERENCE_GO.json',dict(status='FROZEN',models=models,pins=[pin(p) for p in paths],runtime_pins=runtime,core_xywh=g['selected_core_xywh'],grids=g['grids'],centre_depth=0,directions=['forward','reverse'],batch_size=1,threads=4,precision='fp32 CPU',maximum_candidate_forwards=288,read_side_known=False))
print(json.dumps(dict(status='FROZEN',input_files=len(paths),runtime_files=len(runtime),planned_maps=16,maximum_candidate_forwards=288)))
