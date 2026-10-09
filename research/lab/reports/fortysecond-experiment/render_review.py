"""Native-data review packet; no model predictions, annotations or ink inference."""
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from common import *
from alignment import sample,basis
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache');os.environ['XDG_CACHE_HOME']=str(HERE/'plot-cache/xdg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 out=HERE/'review';out.mkdir(exist_ok=True);need(not list(out.iterdir()),'Empty review output');selection=json.loads((HERE/'geometry/SELECTION.json').read_text());receipt=json.loads((HERE/'ct/COMPLETE.json').read_text());meta=json.loads((PRIOR/'PAIR_FEASIBILITY.json').read_text());A=np.asarray(meta['forward']['matrix']);B=np.asarray(meta['inverse']['matrix']);geom=np.load(HERE/'geometry/old-xyz.npy');align=json.loads((HERE/'alignment/RESULTS.json').read_text());figures=[]
 for c in selection['candidates']:
  block={r['volume']:r for r in receipt['arrays'] if r['pair']==c['id']};old=np.load(verify(block['old']['array']));new=np.load(verify(block['new']['array']));smooth=gaussian_filter(new.astype(np.float32),(7.91/2.4)/2,truncate=3)
  oldbasis=basis(geom,c['old_grid_yx']);u=A[:,:3]@oldbasis[0];u/=np.linalg.norm(u);v=A[:,:3]@oldbasis[1];v-=u*np.dot(u,v);v/=np.linalg.norm(v);n=np.cross(u,v);point=np.asarray(c['new_xyz_published_transform']);g=np.arange(-64,64);yy,xx=np.meshgrid(g,g,indexing='ij')
  views=[(f'Surface parallel n={offset*2.4:+.1f}µm',point+xx[...,None]*u+yy[...,None]*v+offset*n) for offset in [-32,0,32]]
  views += [('U/normal cross-section',point+xx[...,None]*u+yy[...,None]*n),('V/normal cross-section',point+xx[...,None]*v+yy[...,None]*n)]
  fitted=next(r for r in align['rows'] if r['id']==c['id'])['fit'];shift=np.asarray(fitted['refined_shift_zyx'])[::-1] if fitted is not None else np.zeros(3)
  scales={k:np.percentile(a[a>0],[1,99.5]).tolist() for k,a in [('old',old),('new',new)]};fig,axes=plt.subplots(4,5,figsize=(14,11.7));coverage=[]
  arrays={};maskarrays={}
  for j,(name,points) in enumerate(views):
   oldpoints=points@B[:,:3].T+B[:,3]
   for i,(tag,source,coords,kind) in enumerate([('H native2.4µm',new,points,'new'),('S fixed blur',smooth,points,'new'),('R public transform',old,oldpoints,'old'),('R fitted translation',old,oldpoints+shift,'old')]):
    image,valid=sample(source,coords,block[kind]['start_zyx']);a=axes[i,j];a.imshow(image,cmap='gray',vmin=scales[kind][0],vmax=scales[kind][1],origin='upper');a.axvline(64,color='#ee6655',alpha=.5,lw=.5);a.axhline(64,color='#ee6655',alpha=.5,lw=.5);a.set_xticks([0,64,127],['−154','0','+151']);a.set_yticks([0,64,127],['−154','0','+151']);a.tick_params(labelsize=7)
    if i==0:a.set_title(name,fontsize=9)
    if j==0:a.set_ylabel(tag if i!=3 or fitted is not None else 'No fit: public transform',fontsize=10)
    coverage.append(dict(row=i,column=j,valid_fraction=float(valid.mean())));arrays[f'r{i}c{j}']=image;maskarrays[f'r{i}c{j}']=valid
  np.savez_compressed(out/(c['id']+'-views.npz'),**arrays);np.savez_compressed(out/(c['id']+'-coverage.npz'),**maskarrays)
  fig.suptitle(c['id']+' · raw acquisition comparison; no ink labels or detector outputs',fontsize=14);fig.text(.015,.018,'Axes are micrometres in the modern-volume basis. Each panel spans307µm. Common display ranges within each acquisition.\nS is a blur visualization; the alignment diagnostic additionally samples it on a coarse lattice. Correlation fits do not verify ink or wrap identity.',fontsize=9)
  fig.tight_layout(rect=(0,.06,1,.96));path=out/(c['id']+'.png');fig.savefig(path,dpi=150);plt.close(fig)
  figures.append(dict(id=c['id'],figure=pin(path),views=pin(out/(c['id']+'-views.npz')),coverage=coverage,display_percentiles=scales,new_basis_uvn=np.stack([u,v,n]).tolist(),old_translation_xyz=shift.tolist()))
 save(out/'FIGURES.json',dict(status='RAW_REVIEW_PACKET_CREATED',created_at_utc=now(),figures=figures,new_model_calls=0,independent_ink_labels_created=0))
 print(json.dumps(dict(status='PASS',figures=len(figures))))
if __name__=='__main__':main()
