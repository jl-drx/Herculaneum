import os
from study_common import *
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache/mpl');os.environ['XDG_CACHE_HOME']=str(HERE/'plot-cache/xdg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=HERE/'figures';out.mkdir(exist_ok=False)
fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
for arm,label in [('A_control','A: BCE + KL'),('B_dice','B: BCE + KL + Dice')]:
    r=json.loads((HERE/'run-v1'/arm/'COMPLETE.json').read_text());steps=[0,*UPDATES];rows=[r['measurements'][str(n)]['mean'] for n in steps]
    for a,key,title in zip(ax.flat,['hard_base','loss_U','dice_loss','loss_D_M','probability_drift_manual','full'],['Common hard-label BCE','Common smoothed BCE','Common negative Dice','Manual teacher KL','Manual probability drift','Own objective change from initialization']):
        values=[row[key]-(rows[0][key] if key=='full' else 0) for row in rows];a.plot(steps,values,'o-',label=label);a.set_title(title);a.set_xlabel('Completed updates');a.grid(alpha=.2);a.axvline(32,color='gray',linestyle=':',linewidth=1)
ax[0,0].legend();fig.suptitle('Full-network budget × objective study: fixed training exposures\nOwn objective changes are not directly comparable losses')
fig.savefig(out/'learning-curves.png',dpi=140);plt.close(fig)
m=json.loads((HERE/'METRICS.json').read_text());roles=['A_control-32','A_control-512','B_dice-32','B_dice-512'];ids=[c['id'] for c in json.loads((HERE/'PLAN.json').read_text())['cases']]
fig,axs=plt.subplots(2,1,figsize=(14,9),layout='constrained')
for ax,key,title in zip(axs,['average_precision','roc_auc'],['AP change versus unchanged seed42','AUROC change versus unchanged seed42']):
    for j,role in enumerate(roles):
        delta=[m['roles'][role]['contexts'][cid][key]-m['roles']['seed42']['contexts'][cid][key] for cid in ids];ax.bar(np.arange(8)+(j-1.5)*.19,delta,width=.19,label=role)
    ax.axhline(0,color='black',linewidth=.7);ax.axhline(-.02,color='red',linestyle='--',label='Regional retention floor');ax.set_xticks(range(8),[s.replace('pherc0814-46527-fresh-','0814-') for s in ids],rotation=25,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
axs[0].legend(ncol=3,fontsize=9);fig.suptitle('Every region is reused development; scores measure supplied-label agreement')
fig.savefig(out/'all-regional-deltas.png',dpi=140);plt.close(fig)
save(HERE/'FIGURES.json',dict(generator=pin(__file__),figures=[pin(x) for x in out.glob('*.png')]))
