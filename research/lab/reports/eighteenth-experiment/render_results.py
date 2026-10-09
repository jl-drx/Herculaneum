"""Render fixed, completed metrics without new predictions or selection."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/herculaneum-phase18-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/herculaneum-phase18-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
def record(p):
    p=Path(p).resolve();return dict(path=str(p),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
source=HERE/'analysis/metrics.json';assert record(source)['sha256']=='d2995ff521a87255ad0cb2d991ebc15f0134f4cd7b84e1278361c3d13dffd662'
d=json.loads(source.read_text());out=HERE/'figures';out.mkdir(exist_ok=False)
fig,ax=plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
colors={'raw_control':'#b07535','explicit_depth':'#286d91'}
for arm in colors:
    t=d['training'][arm];steps=[0,1,8,16,32]
    for key,style in [('full','-'),('base','--')]:
        values=[0]+[t['fixed_updates'][str(s)][key]-t['initial'][key] for s in steps[1:]]
        ax[0].plot(steps,values,style,marker='o',markersize=4,color=colors[arm],
                   label=arm.replace('_',' ')+' / '+key)
ax[0].axhline(0,color='gray',linewidth=.8)
ax[0].set(xlabel='Completed updates',ylabel='Fixed training-risk change from initializer',
          title='Depth features mainly reduce prediction drift',xticks=[0,1,8,16,32])
ax[0].legend(fontsize=8,loc='upper right')
ids=sorted(d['per_case']);positions=np.arange(len(ids))
for delta,key,color in [(-.16,'auc','#286d91'),(.16,'ap','#4a8a63')]:
    values=[d['per_case'][c]['models']['explicit_depth'][key]-d['per_case'][c]['models']['raw_control'][key] for c in ids]
    ax[1].barh(positions+delta,values,height=.29,label=key.upper(),color=color)
ax[1].axvline(0,color='gray',linewidth=.8)
ax[1].set(yticks=positions,yticklabels=[c.replace('frag','F').replace('-context',' / C') for c in ids],
          xlabel='Explicit depth minus matched raw control',title='Small, uneven development ranking changes')
ax[1].set_ylim(6,-.6)
ax[1].legend(loc='lower right',fontsize=8)
ax[0].text(0,-.2,'Same PHerc0139 data, initialization and 32 updates.\nFull = smoothed manual BCE + preservation KL.',transform=ax[0].transAxes,fontsize=8)
ax[1].text(0,-.2,'All six reused development regions shown; no fresh validation.\nFeature screen failed; unchanged seed42 remains the reference.',transform=ax[1].transAxes,fontsize=8)
fig.suptitle('Plinius: matched 801-parameter residual experiment',fontsize=14)
fig.savefig(out/'depth-residual-pilot.png',dpi=180)
fig.savefig(out/'depth-residual-pilot.svg')
plt.close(fig)
receipt=dict(status='PASS',source=record(__file__),metrics=record(source),outputs=[record(out/n) for n in ['depth-residual-pilot.png','depth-residual-pilot.svg']],new_model_calls=0)
(out/'presentation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='PASS',receipt=record(out/'presentation-receipt.json'))))
