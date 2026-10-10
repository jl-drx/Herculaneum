"""Render every fixed region and shift; no inference or variant selection."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache/matplotlib')
os.environ['XDG_CACHE_HOME']=str(HERE/'plot-cache/xdg')
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
j=json.loads((HERE/'run-v1/COMPLETE.json').read_text());cs=j['contexts'];out=HERE/'figures-v2';out.mkdir(exist_ok=False)
roles=['baseline','candidate','optical_darkness'];names=['Unchanged CT baseline','Fixed CT averaging','Optical darkness']
for family,title,note in [('common','All shifts on common known support','Common support removes most shifting boundaries; constant scores can be uninformative.'),
    ('moving_support','All shifts on each shift’s own support','Coverage-confounded: displacement, support and prevalence change together.\nNo shift selected.')]:
    fig,axs=plt.subplots(4,3,figsize=(13,14))
    fig.subplots_adjust(left=.065,right=.99,top=.91,bottom=.04,hspace=.60,wspace=.30)
    for row,c in enumerate(cs):
        for col,(role,name) in enumerate(zip(roles,names)):
            ax=axs[row,col];v=np.array([x['metrics'][role]['ap'] for x in c[family+'_shifts']]).reshape(3,3)
            ax.imshow(v,cmap='viridis',vmin=0,vmax=1)
            for y in range(3):
                for x in range(3):ax.text(x,y,f'{v[y,x]:.4f}',ha='center',va='center',color='white' if v[y,x]<.65 else 'black',fontsize=11)
            ax.set_xticks(range(3),['−1','0','+1']);ax.set_yticks(range(3),['−1','0','+1']);ax.set_xlabel('dx (coarse raster pixels)');ax.set_ylabel('dy')
            s=c['support'];suffix=f"common {s['common_known']}; changeable {s['potentially_changeable_common']}" if family=='common' else 'variable known support'
            ax.set_title(f"{c['id']} · {name}\n{suffix}",fontsize=9)
    fig.suptitle(title+' — average precision (AP)\n'+note,fontsize=13)
    fig.savefig(out/(family+'-shift-ap.png'),dpi=150);plt.close(fig)
fig,axs=plt.subplots(4,5,figsize=(15,12),layout='constrained')
for row,c in enumerate(cs):
    a=np.load(c['arrays']['path'],allow_pickle=False);labels=a['shifted_labels'][4]
    band=np.zeros((64,64),np.uint8)
    for i in range(3):band[a['band_masks'][i]]=i+1
    panels=[(1-a['predictions'][2],'gray',0,1),(labels,ListedColormap(['#ededed','#202020','#eeb833']),0,2),
            (a['predictions'][0],'magma',0,1),(a['predictions'][1],'magma',0,1),
            (band,ListedColormap(['#eeb833','#4477aa','#66ccee','#228833']),0,3)]
    for col,(v,cmap,lo,hi) in enumerate(panels):
        ax=axs[row,col];ax.imshow(v,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        if row==0:ax.set_title(['Optical mean intensity','Supplied labels','CT baseline probability','CT averaging probability','Distance bands'][col])
        if col==0:ax.set_ylabel(c['id'])
fig.suptitle('All four 64 × 64 scoring cores · fixed probability scale 0–1\nLabels: dark = ink, light = clear, yellow = unknown. Bands: blue ≤2, cyan 2–4, green >4, yellow unknown.\nOptical and labels may share provenance; these panels do not authenticate ink.',fontsize=12)
fig.savefig(out/'all-context-panels.png',dpi=150);plt.close(fig)
print(json.dumps({'status':'PASS','figures':[str(p) for p in sorted(out.glob('*.png'))]}))
