import os
from common_v2 import *
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache/mpl');os.environ['XDG_CACHE_HOME']=str(HERE/'plot-cache/xdg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=HERE/'figures';out.mkdir(exist_ok=False)
r=json.loads((HERE/'run-v1/C_coverage/COMPLETE.json').read_text());steps=[0,*UPDATES];rows=[r['measurements'][str(s)]['mean'] for s in steps]
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for a,key,title in zip(axes,['full','hard_base','dice_loss'],['Optimized BCE + KL','Hard-label BCE','Negative soft Dice (diagnostic)']):
    a.plot(steps,[row[key] for row in rows],'o-',color='teal');a.set_title(title);a.set_xlabel('Completed updates');a.grid(alpha=.2)
fig.suptitle('Mixed-schedule training risk: fixed class exposure across two parents')
fig.savefig(out/'learning-curves.png',dpi=140);plt.close(fig)
m=json.loads((HERE/'METRICS.json').read_text());ids=[c['id'] for c in json.loads((HERE/'PLAN-V2.json').read_text())['cases']]
fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained')
for ax,key,title in zip(axes,['average_precision','roc_auc'],['AP change versus unchanged seed42','AUROC change versus unchanged seed42']):
    for j,(role,label) in enumerate([('A_control-512','Original two-region control'),('C_coverage-512','Exposure-matched two-parent mixture')]):
        values=[m['roles'][role]['contexts'][cid][key]-m['roles']['seed42']['contexts'][cid][key] for cid in ids];ax.bar(np.arange(8)+(j-.5)*.35,values,width=.35,label=label)
    ax.axhline(0,color='black',linewidth=.7);ax.axhline(-.02,color='red',linestyle='--',label='Retention floor');ax.set_xticks(range(8),[s.replace('pherc0814-46527-fresh-','0814-') for s in ids],rotation=25,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=9);fig.suptitle('All eight regions are reused development; supplied-label agreement')
fig.savefig(out/'all-regional-deltas.png',dpi=140);plt.close(fig)
save(HERE/'FIGURES.json',dict(generator=pin(__file__),figures=[pin(p) for p in out.glob('*.png')]))
