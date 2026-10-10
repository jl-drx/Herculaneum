from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
P=Path(__file__).resolve().parent;W=P/'work';A=P/'independent-audit';r=json.loads((P/'INFERENCE_RESULTS.json').read_text())
models=['dnative','hybrid_3d2d-seed42','hybrid_3d2d-seed43','hecate'];cols=[('plain','forward'),('plain','reverse'),('padded','forward'),('padded','reverse')]
fig,axes=plt.subplots(4,4,figsize=(9,9),layout='constrained')
for i,name in enumerate(models):
 for j,(phase,direction) in enumerate(cols):
  rec=next(r for r in r['maps'] if (r['model'],r['phase'],r['direction'])==(name,phase,direction));a=np.load(rec['outputs']['display']['path'])
  axes[i,j].imshow(a,cmap='gray',vmin=0,vmax=255,interpolation='nearest');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
  if i==0:axes[i,j].set_title(phase+' / '+direction)
  if j==0:axes[i,j].set_ylabel(name.replace('hybrid_3d2d-',''),rotation=90)
fig.suptitle('PHerc0800 segment225813 — all16 fixed-scale responses\nx1024:1152,y280:408; d0; no fitted contrast or reading claim')
fig.savefig(P/'ALL_MAPS.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(11,4),layout='constrained');m=np.load(W/'candidate-admission-masks.npz')
for axis,key,title in [(ax[0],'current_archived_plain_villa','Proposed192-square: plain Villa'),(ax[1],'common_current','Joint grids/phases: 128-square retained')]:
 axis.imshow(m[key],cmap='gray',vmin=0,vmax=1,extent=[1024,1216,448,256]);axis.add_patch(Rectangle((1024,280),128,128,fill=False,color='#ee7733',lw=2));axis.set_title(title);axis.set_xlabel('Source x');axis.set_ylabel('Source y')
a=np.load(W/'candidate-render66.npy');# Mosaic origin952,192; centre layer32 is mesh-0.5voxel.
ax[2].imshow(a[32,88:216,72:200],cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax[2].set_title('Admitted core CT, layer32\nFixed0..255, not an ink label');ax[2].set_xticks([]);ax[2].set_yticks([])
fig.suptitle('Independent fixed-grid lead → full input-support admission')
fig.savefig(P/'CANDIDATE_SUPPORT.png',dpi=160);plt.close(fig)
