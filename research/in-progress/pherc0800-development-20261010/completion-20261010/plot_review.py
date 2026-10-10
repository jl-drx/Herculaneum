"""Every fixed panel at0..255, including the entire3x3 partition; no selection."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;W=P/'work';O=P/'review-panels';O.mkdir()
r=json.loads((P/'INFERENCE_RESULTS.json').read_text());assert json.loads((P/'COMPLETION_VERIFICATION.json').read_text())['status']=='PASS'
mask=np.load(W/'output-support.npz')['common'];models=['dnative','hybrid_3d2d-seed42','hybrid_3d2d-seed43','hecate'];cols=[('plain','forward'),('plain','reverse'),('padded','forward'),('padded','reverse')]
arrays={(m['model'],m['phase'],m['direction']):np.load(m['outputs']['aligned']['path'])*255 for m in r['maps']};cmap=plt.get_cmap('gray').copy();cmap.set_bad('#93655a')
def plot(path,x,y,w,h,title,dpi):
 fig,axes=plt.subplots(4,4,figsize=(14,14),layout='constrained');sl=np.s_[y:y+h,x:x+w]
 for i,model in enumerate(models):
  for j,(phase,direction) in enumerate(cols):
   a=np.ma.array(arrays[model,phase,direction][sl],mask=~mask[sl]);axes[i,j].imshow(a,cmap=cmap,vmin=0,vmax=255,interpolation='nearest');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
   if i==0:axes[i,j].set_title(phase+' / '+direction)
   if j==0:axes[i,j].set_ylabel(model.replace('hybrid_3d2d-',''))
 fig.suptitle(title+'\nAll16 fixed-scale0..255 responses; brown = unsupported, not observed zero')
 fig.savefig(path,dpi=dpi);plt.close(fig)
plot(P/'ALL_MAPS.png',0,0,1536,1536,'PHerc0800 segment20251028222030; source x512:2048/y512:2048; d0',180)
rows=[]
for iy in range(3):
 for ix in range(3):
  x,y=ix*512,iy*512;name=f'region-{iy+1}-{ix+1}.png';plot(O/name,x,y,512,512,f'Region{iy+1},{ix+1}: source x{x+512}:{x+1024}/y{y+512}:{y+1024}',150);rows.append(dict(file=name,source_xywh=[x+512,y+512,512,512],common_pixels=int(mask[y:y+512,x:x+512].sum())))
(O/'INDEX.json').write_text(json.dumps(dict(regions=rows,selection='Complete predetermined3x3partition, not chosen by detector response'),indent=2)+'\n')
ct=np.load(W/'render66.npy',mmap_mode='r')[32,512:2048,512:2048];known=np.load(W/'observed-mask.npy')[512:2048,512:2048]
fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained');ax[0].imshow(np.ma.array(ct,mask=~known),cmap=cmap,vmin=0,vmax=255,interpolation='nearest');ax[0].set_title('Native CT plane32 (mesh -0.5voxel)')
ax[1].imshow(known,cmap='gray',vmin=0,vmax=1);ax[1].set_title('Observed geometry/input footprint');ax[2].imshow(mask,cmap='gray',vmin=0,vmax=1);ax[2].set_title('All-model/phase context support:79.50%')
for a in ax:a.set_xticks([]);a.set_yticks([])
fig.savefig(P/'CT_AND_SUPPORT.png',dpi=160);plt.close(fig)
print('All16 overview,9 complete regional panels and CT/support plot written')
