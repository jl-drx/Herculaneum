from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;r=json.loads((P/'support-results/RESULTS.json').read_text())
fig,axes=plt.subplots(2,5,figsize=(16,7),layout='constrained')
for i,row in enumerate(r['meshes']):
 z=np.load(P/'support-results'/f"{row['segment']}.npz")
 stage=np.zeros(z['bilinear_valid'].shape,np.uint8);stage[z['bilinear_valid']]=1;stage[z['normal_valid']]=2
 stage[z['inside_CT']&~z['old_surface_near']]=3;stage[z['point_support']]=4;stage[z['context_core']]=5
 axes[0,i].imshow(stage,vmin=0,vmax=5,cmap=matplotlib.colors.ListedColormap(['black','#8a8a8a','#d67b32','#9581bf','#54a6b5','#ffe053']),interpolation='nearest')
 axes[0,i].set_title(row['segment'],fontsize=9)
 axes[1,i].imshow(z['context_core'],vmin=0,vmax=1,cmap='gray',interpolation='nearest')
 b=row['unrestricted_best_square']['box_xywh']
 if b:
  x,y,w,h=b;axes[1,i].add_patch(plt.Rectangle((x,y),w,h,fill=False,color='red',linewidth=1))
 axes[1,i].set_title('Largest square: '+str(row['unrestricted_best_square']['side'])+'px',fontsize=9)
 for a in axes[:,i]:a.set_xticks([]);a.set_yticks([])
fig.suptitle('PHerc0800 geometry only — top: successive support; bottom: full-context cores\nBlack unknown; grey base only; orange valid normals; purple surface-separated; cyan also CT-box-separated; yellow full-context',fontsize=11)
fig.savefig(P/'SUPPORT.png',dpi=125);plt.close(fig)
print('Support figure written')
