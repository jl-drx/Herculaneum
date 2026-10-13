"""Freeze minimal grid contexts and bounded fullcanvas-aligned renderer acquisition."""
from pathlib import Path
import json,itertools,shutil
import numpy as np,tifffile
from admit_candidate import pin,save,starts
from support import reconstruct,corner_envelope
P=Path(__file__).resolve().parent;W=P/'work';R=P.parent;LAB=P.parents[1]
d=json.loads((P/'ADMISSION.json').read_text());assert d['selected_core_xywh'] is not None
S=d['candidate_segment'];G=LAB/'data/experiments/pherc0800-larger-geometry'/S
cx,cy,cw,ch=d['selected_core_xywh'];shape=[2300,2260];models=[];bounds=[]
for phase,py,px in [('plain',0,0),('padded',40,200)]:
 h,w=shape[0]+py,shape[1]+px
 for family,p,stride in [('villa',128,64),('hecate',64,32)]:
  hh,ww=(h,w) if family=='villa' else (round(h*.9),round(w*.9))
  ylo,yhi,xlo,xhi=cy+py,cy+py+ch,cx+px,cx+px+cw
  if family=='hecate':
   ylo,yhi=int(np.floor((ylo+.5)*hh/h-.5)),int(np.floor((yhi-.5)*hh/h-.5))+2
   xlo,xhi=int(np.floor((xlo+.5)*ww/w-.5)),int(np.floor((xhi-.5)*ww/w-.5))+2
  patches=[[x,y,p,p] for y in starts(hh,p,stride) for x in starts(ww,p,stride) if y<yhi and y+p>ylo and x<xhi and x+p>xlo]
  y0=min(a[1] for a in patches);y1=max(a[1]+p for a in patches);x0=min(a[0] for a in patches);x1=max(a[0]+p for a in patches)
  if family=='hecate':y0,y1=int(np.floor(y0*h/hh))-1,int(np.ceil(y1*h/hh))+1;x0,x1=int(np.floor(x0*w/ww))-1,int(np.ceil(x1*w/ww))+1
  box=[x0-px,y0-py,x1-px,y1-py];bounds.append(box)
  models.append(dict(phase=phase,family=family,grid_shape_yx=[hh,ww],patches_xywh=patches,patch_count=len(patches),source_context_bbox_xyxy_bound=box))
x0=min(b[0] for b in bounds)//128*128;y0=min(b[1] for b in bounds)//128*128
x1=(max(b[2] for b in bounds)+127)//128*128;y1=(max(b[3] for b in bounds)+127)//128*128
mesh=np.stack([tifffile.imread(G/f'mesh.{a}.tif') for a in 'xyz'],-1)
b,n,bv,nv=reconstruct(mesh,json.loads((G/'mesh.meta.json').read_text())['scale'],shape)
sl=np.s_[y0:y1,x0:x1];assert bv[sl].all() and nv[sl].all(), 'Do not silently use unknown extra renderer pixels'
np.savez_compressed(W/'render-geometry.npz',base=b[sl],normal=n[sl],valid=nv[sl])
lo,hi=corner_envelope(b[sl],n[sl]);lo=lo.astype(int)//128;hi=hi.astype(int)//128;assert (hi-lo<=1).all()
keys=set()
for delta in itertools.product([0,1],repeat=3):
 v=lo+delta;keep=(v<=hi).all(-1)
 for xyz in np.unique(v[keep],axis=0):keys.add('0/'+'/'.join(map(str,xyz[::-1])))
assert len(keys)*128**3<=128*1024**2
p49=json.loads((R/'fortyninth-experiment/GO.json').read_text());oldkeys=p49['source_chunk_keys']
retained=[]
for key in oldkeys:
 f=R/'fortyninth-experiment/work/source-cache'/key;retained.append(dict(key=key,**pin(f)))
for seg,label in [(S,'candidate'),('20251028222030','metadata-control')]:
 dest=W/(label+'.tifxyz');dest.mkdir()
 src=LAB/'data/experiments/pherc0800-larger-geometry'/seg
 for name in ['x.tif','y.tif','z.tif','meta.json']:shutil.copyfile(src/f'mesh.{name}',dest/name)
plan=dict(status='FROZEN',protocol=pin(P/'PROTOCOL.md'),script=pin(__file__),admission=pin(P/'ADMISSION.json'),geometry=pin(W/'render-geometry.npz'),source_url=p49['source_url'],selected_core_xywh=[cx,cy,cw,ch],candidate_render_xywh=[x0,y0,x1-x0,y1-y0],candidate_source_keys=sorted(keys),metadata_control_source_keys=oldkeys,source_keys=sorted(keys|set(oldkeys)),retained_chunks=retained,new_candidate_payload_upper_bound_bytes=len(keys-set(oldkeys))*128**3,all_render_geometry_valid=True,extra_render_pixels_are_not_admitted_core=True,grids=models,cases=[dict(name='metadata-control',segmentation=str(W/'metadata-control.tifxyz'),crop_xywh=[1216,1216,128,128],source_keys=oldkeys),dict(name='candidate',segmentation=str(W/'candidate.tifxyz'),crop_xywh=[x0,y0,x1-x0,y1-y0],source_keys=sorted(keys))],voxel_size_um=8.64,depth=66,offsets=[-32.5,32.5],flip_normals=True)
save(P/'RENDER_GO.json',plan)
print(json.dumps({k:plan[k] for k in ['selected_core_xywh','candidate_render_xywh','new_candidate_payload_upper_bound_bytes','all_render_geometry_valid','grids']}))
