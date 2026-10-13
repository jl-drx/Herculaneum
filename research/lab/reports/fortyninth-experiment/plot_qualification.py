from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;W=P/'work'
valid=np.load(W/'target-planning-masks.npz')['valid_geometry'][512:2048,512:2048]
common=np.load(W/'planned-output-support.npz')['common']
a=np.load(W/'render31.npy');b=np.load(W/'published31-crop.npy')
fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
ax[0].imshow(valid,cmap='gray',vmin=0,vmax=1);ax[0].set_title('Original-mesh geometric support\n92.8% of existing field')
ax[1].imshow(common,cmap='gray',vmin=0,vmax=1);ax[1].set_title('All four input-grid/phase supports\n79.5%; planned, not fully rendered')
ax[2].imshow(a[15],cmap='gray',vmin=0,vmax=255);ax[2].set_title('Fixed qualification patch, centre plane\nCT intensity, not an ink prediction')
for panel in ax:panel.set_xticks([]);panel.set_yticks([])
fig.suptitle('PHerc0800 development transfer qualification — study49')
fig.savefig(P/'QUALIFICATION.png',dpi=150);plt.close(fig)
