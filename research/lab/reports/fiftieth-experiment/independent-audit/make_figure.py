from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
p=Path(__file__).parent;d=json.loads((p/'output/independent-results.json').read_text())
fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained')
for ax,row in zip(axs.flat,d['meshes']):
 s=row['segment'];a=np.load(p/'output'/f'{s}-masks.npz');h,w=a['point_support'].shape
 rgb=np.full((h,w,3),255,np.uint8);rgb[a['bilinear_valid']]=[220,222,225];rgb[a['point_support']]=[80,145,200];rgb[a['fixed_grid_context']]=[242,170,65];rgb[a['all_origin_context']]=[35,145,90]
 ax.imshow(rgb,origin='upper',interpolation='nearest');ax.set_title(s+'\nConservative / fixed-grid area: '+format(row['counts']['all_origin_context'],',')+' / '+format(row['counts']['fixed_grid_context'],','),fontsize=10)
 ax.set_xlabel('Native source x (pixels)');ax.set_ylabel('Native source y (pixels)')
 if s=='20251028225813': ax.add_patch(Rectangle((1024,256),192,192,fill=False,edgecolor='black',lw=1.5))
ax=axs.flat[-1];a=np.load(p/'output/20251028225813-masks.npz');rgb=np.full((*a['point_support'].shape,3),255,np.uint8);rgb[a['bilinear_valid']]=[220,222,225];rgb[a['point_support']]=[80,145,200];rgb[a['fixed_grid_context']]=[242,170,65];rgb[a['all_origin_context']]=[35,145,90]
ax.imshow(rgb,origin='upper',interpolation='nearest');ax.set_xlim(880,1380);ax.set_ylim(590,100);ax.add_patch(Rectangle((960,192),320,320,fill=False,edgecolor='purple',lw=2,ls='--'));ax.add_patch(Rectangle((1024,256),192,192,fill=False,edgecolor='black',lw=2));ax.add_patch(Rectangle((1093,318),69,69,fill=False,edgecolor='white',lw=1.5));ax.set_title('25813 / component 1: conditional 192×192 core\n16 covering patches; 320×320 source union',fontsize=11);ax.set_xlabel('Native source x (pixels)');ax.set_ylabel('Native source y (pixels)')
fig.suptitle('PHerc0800: conservative gate fails; one exact native grid preserves a review candidate\nGeometry and exposure support only. No ink evidence or full-recipe admission.',fontsize=15)
fig.legend(handles=[Patch(facecolor=[220/255,222/255,225/255],label='Bilinear valid'),Patch(facecolor=[80/255,145/255,200/255],label='Point support'),Patch(facecolor=[242/255,170/255,65/255],label='Exact origin (0,0), patch 128 / stride 64'),Patch(facecolor=[35/255,145/255,90/255],label='Conservative radius 128')],loc='outside lower center',ncol=4,fontsize=10)
fig.savefig(p/'SUPPORT_COMPARISON.png',dpi=160);plt.close(fig)
