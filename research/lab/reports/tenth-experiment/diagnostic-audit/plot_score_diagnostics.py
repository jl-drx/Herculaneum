"""Fixed-scale display of retained descriptive score quantiles and cutoffs."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'result/score-diagnostics.json'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='399c03273d127d7a53c2d3b95e9bf726f88107752b970b5be23b5e4f8f9952f4'
data=json.loads(SOURCE.read_text())
arms=['frozen_seed42','A','B','C','A_smooth','B_smooth']
cases=[(key,value) for key,value in data['contexts'].items() if value['role']=='diagnostic']
fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharex=True,sharey=True,layout='constrained')
for ax,(cid,case) in zip(axes,cases):
    for index,arm in enumerate(arms):
        y=len(arms)-1-index;m=case['models'][arm]
        for label,offset,color in [('positive',.16,'#c45e10'),('negative',-.16,'#146a9e')]:
            q=m['class_score_distribution'][label]['quantiles']
            ax.hlines(y+offset,q['q05'],q['q95'],color=color,lw=3,label=label+' q05–q95' if index==0 else None)
            ax.plot(q['median'],y+offset,'o',color=color,ms=5)
        ax.plot(m['operating']['own_primary']['threshold'],y,'v',color='black',ms=6,label='existing own primary cutoff' if index==0 else None)
    ax.axvline(data['existing_thresholds']['frozen_seed42']['primary'],color='#555555',ls='--',lw=1,label='existing frozen numeric cutoff')
    ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks(list(range(len(arms))),list(reversed(arms)))
    ax.set_ylim(-.6,len(arms)-.4);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Raw score (uncalibrated)')
    ax.set_title(cid.removeprefix('pherc0814-46527-fresh-')+' · '+str(case['positive_count'])+' positive / '+str(case['negative_count'])+' negative')
axes[0].legend(loc='upper center',bbox_to_anchor=(1.03,-.14),ncol=2,frameon=False,fontsize=9)
fig.suptitle('Retained supplied-annotation score distributions\nCentral 90% intervals; circles = medians; triangles = already locked cutoffs',fontsize=12)
target=HERE/'result/class-score-intervals.png'
assert not target.exists()
fig.savefig(target,dpi=160);plt.close(fig)
print(json.dumps({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
