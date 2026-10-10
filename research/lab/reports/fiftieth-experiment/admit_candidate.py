"""Exact model-grid footprint admission inside the VM's single proposed core."""
from pathlib import Path
import hashlib,json,sys,itertools
import numpy as np,cv2,tifffile
P=Path(__file__).resolve().parent;R=P.parent;LAB=P.parents[1];A=P/'independent-audit';W=P/'work';W.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'fortyeighth-experiment'));from support import reconstruct,corner_envelope,largest_square
S='20251028225813';G=LAB/'data/experiments/pherc0800-larger-geometry'/S

def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h)
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def starts(n,p,s):return sorted(set(list(range(0,max(1,n-p+1),s))+[max(0,n-p)]))
def main():
 assert not (P/'GO.json').exists()
 prior=R/'fortyninth-experiment/final-verification.json';assert pin(prior)['sha256']=='0e864a6e95214235ee1d13c28cd6b415d8f0a51f497503fb1bab6972e0eef618'
 old=R/'fortyeighth-experiment';g=json.loads((old/'GO.json').read_text())
 pins=[pin(P/'PROTOCOL.md'),pin(__file__),pin(P/'independent-audit.zip'),pin(old/'support.py'),pin(old/'support-results'/f'{S}.npz'),pin(A/'output-locked'/f'{S}-masks.npz'),pin(A/'candidate.json'),pin(R/'fortyninth-experiment/GO.json')]
 for name in ['mesh.x.tif','mesh.y.tif','mesh.z.tif','mesh.meta.json','surface.level0.zarray.json','source-render-export-metadata.json']:
  p=G/name;r=pin(p);expected=next(x for x in g['pins'] if x['path']==str(p));assert r==expected;pins.append(r)
 save(P/'GO.json',dict(prior_seal=pin(prior),pins=pins,candidate_segment=S,candidate_core_xywh=[1024,256,192,192],minimum_subcore=128,padding_top_left=[40,200],candidate_new_raw_byte_cap=128*1024**2,new_intensity_or_inference_access=False))
 mesh=np.stack([tifffile.imread(G/f'mesh.{a}.tif') for a in 'xyz'],-1);shape=json.loads((G/'surface.level0.zarray.json').read_text())['shape'][1:];assert shape==[2300,2260]
 b,n,bv,nv=reconstruct(mesh,json.loads((G/'mesh.meta.json').read_text())['scale'],shape)
 oldm=np.load(old/'support-results'/f'{S}.npz');other=np.load(A/'output-locked'/f'{S}-masks.npz')
 historic_archived=oldm['point_support'];historic_independent=other['point_support']
 # Check new whole raw-chunk exposure since the original frozen inventory.
 newkeys=json.loads((R/'fortyninth-experiment/GO.json').read_text())['source_chunk_keys'];oldkeys=set(map(tuple,g['raw_native_chunk_indices']))
 additions=[];newhit=np.zeros(shape,bool)
 lo,hi=corner_envelope(b,n)
 for key in newkeys:
  zyx=tuple(map(int,key.split('/')[1:]))
  if zyx in oldkeys:continue
  lower=np.array(zyx[::-1])*128;upper=lower+127
  hit=nv&(hi>=lower).all(-1)&(lo<=upper).all(-1);newhit|=hit
  additions.append(dict(source_key=key,lo_xyz=lower.tolist(),hi_xyz_inclusive=upper.tolist(),intersecting_geometry_pixels=int(hit.sum()),intersecting_historic_point_support=int((hit&historic_archived).sum())))
 # A separate independent arithmetic envelope is needed for the new boxes too.
 import importlib.util
 spec=importlib.util.spec_from_file_location('audit_independent',A/'audit_locked.py');vm=importlib.util.module_from_spec(spec);spec.loader.exec_module(vm)
 independent_newhit=np.zeros(shape,bool)
 for y in range(0,shape[0],128):
  for x in range(0,shape[1],128):
   ib,inn,iv,inv=vm.tile(mesh,json.loads((G/'mesh.meta.json').read_text())['scale'],x,y,min(128,shape[1]-x),min(128,shape[0]-y))
   il=np.floor(ib.astype(float)-32.5*np.abs(inn.astype(float))-.01);ih=np.floor(ib.astype(float)+32.5*np.abs(inn.astype(float))+.01)+1
   a=np.zeros(iv.shape,bool)
   for box in additions:a |= iv&inv&(ih>=box['lo_xyz']).all(-1)&(il<=box['hi_xyz_inclusive']).all(-1)
   independent_newhit[y:y+a.shape[0],x:x+a.shape[1]]=a
 masks=dict(historical_archived=historic_archived,historical_independent=historic_independent,current_archived=historic_archived&~newhit,current_independent=historic_independent&~independent_newhit)
 rows=[];outputs={};context={};cx,cy,cw,ch=1024,256,192,192
 for convention,point in masks.items():
  common=np.ones((ch,cw),bool)
  for phase,py,px in [('plain',0,0),('padded',40,200)]:
   padded=np.pad(point,((py,0),(px,0)),constant_values=False);h,w=padded.shape
   for family,patch,stride in [('villa',128,64),('hecate',64,32)]:
    hh,ww=(h,w) if family=='villa' else (round(h*.9),round(w*.9))
    known=padded if family=='villa' else cv2.resize(padded.astype(np.float32),(ww,hh),interpolation=cv2.INTER_AREA)>=1.0
    integral=np.pad((~known).astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    bad=np.zeros((hh,ww),bool);coverage=np.zeros_like(bad);patches=[]
    # Include both nonzero linear resize neighbors before native ROI cropping.
    ylo,yhi,xlo,xhi=cy+py,cy+py+ch,cx+px,cx+px+cw
    if family=='hecate':
     ylo,yhi=int(np.floor((ylo+.5)*hh/h-.5)),int(np.floor((yhi-.5)*hh/h-.5))+2
     xlo,xhi=int(np.floor((xlo+.5)*ww/w-.5)),int(np.floor((xhi-.5)*ww/w-.5))+2
    for y in starts(hh,patch,stride):
     if y>=yhi or y+patch<=ylo:continue
     for x in starts(ww,patch,stride):
      if x>=xhi or x+patch<=xlo:continue
      count=int(integral[y+patch,x+patch]-integral[y,x+patch]-integral[y+patch,x]+integral[y,x])
      coverage[y:y+patch,x:x+patch]=True
      if count:bad[y:y+patch,x:x+patch]=True
      patches.append(dict(model_xywh=[x,y,patch,patch],unknown_input_pixels=count))
    if family=='hecate':
     bad=cv2.resize(bad.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>0
     coverage=cv2.resize(coverage.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>=1.0
    good=(~bad&coverage)[cy+py:cy+py+ch,cx+px:cx+px+cw];common&=good
    outputs[f'{convention}_{phase}_{family}']=good
    rows.append(dict(convention=convention,phase=phase,family=family,grid_yx=[hh,ww],patch_count=len(patches),fully_supported_patches=sum(r['unknown_input_pixels']==0 for r in patches),supported_core_pixels=int(good.sum()),whole_core_pass=bool(good.all()),patches=patches))
  outputs[convention+'_common']=common
 common=outputs['current_archived_common']&outputs['current_independent_common'];best=largest_square(common)
 if best['box_xywh'] is not None:best['box_xywh'][0]+=cx;best['box_xywh'][1]+=cy
 selected=[cx,cy,cw,ch] if common.all() else best['box_xywh'] if best['side']>=128 else None
 np.savez_compressed(W/'candidate-admission-masks.npz',**outputs,common_current=common,new_chunk_intersection_archived=newhit,new_chunk_intersection_independent=independent_newhit)
 result=dict(status='PASS',decision='CORE_ADMITTED_FOR_RENDER_QUALIFICATION' if selected else 'FULL_RECIPE_CORE_NOT_ADMITTED',candidate_segment=S,proposed_core_xywh=[cx,cy,cw,ch],original_core_pass=bool(common.all()),largest_supported_subcore=best,selected_core_xywh=selected,new_exposure_boxes=additions,rows=rows,current_both_arithmetic_support_pixels=int(common.sum()),mask=pin(W/'candidate-admission-masks.npz'),no_new_intensities_or_model_outputs=True)
 save(P/'ADMISSION.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
