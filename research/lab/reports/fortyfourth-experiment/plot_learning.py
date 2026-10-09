"""Compact scientific figures; fixed metrics, no new model or selection."""
import os
os.environ['MPLCONFIGDIR']='/tmp/herculaneum-mpl';os.environ['XDG_CACHE_HOME']='/tmp/herculaneum-cache'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pixel_http import *
a=json.loads((P/'LEARNING_ANALYSIS.json').read_text());r=json.loads((P/'TRAINING_RESULTS.json').read_text());fig,axs=plt.subplots(1,3,figsize=(12,4.4),dpi=120)
keys=['native_native','native_coarse','coarse_native','coarse_coarse'];names=['1→1','1→4','4→1','4→4'];colors=['#225ea8','#9ecae1','#fdae6b','#e6550d']
for ax,row in zip(axs,a['parents']):
 vals=[row[k]['ap'] for k in keys];ax.bar(names,vals,color=colors);ax.axhline(row['native_native']['prevalence'],color='black',ls=':',label='Constant');best=max(c['metrics']['ap'] for c in row['controls']);ax.axhline(best,color='#666666',ls='--',label='Best scalar control (descriptive)');ax.set_ylim(0,1);ax.set_title(row['held_out']);ax.set_ylabel('Average precision');ax.set_xlabel('Training → test observation factor')
 for i,v in enumerate(vals):ax.text(i,v+.018,f'{v:.4f}',ha='center',fontsize=9)
axs[0].legend(fontsize=7,loc='upper left');fig.suptitle('Three parent holdouts: annotation agreement, not CT transfer');fig.text(.5,.015,'One crop per parent; no uncertainty interval from pixel counts • Vesuvius Challenge2026, CC BY-NC4.0',ha='center',fontsize=8);fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(P/'held-out-ranking.png');plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(12,7),dpi=110)
for arm in r['results']:
 held=int(arm['arm'][4]);q=0 if arm['train_quality']=='native' else 1;ax=axs[q,held]
 for case in arm['train_cases']:
  vals=[next(c['metrics']['ap'] for c in h['cases'] if c['case']==case) for h in arm['training_fit']];ax.plot([512,1024,2048],vals,marker='o',label=case)
 ax.axhline(.8,color='black',ls=':',label='Frozen fit requirement');ax.set_ylim(0,1);ax.set_title(arm['arm']);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Training AP');ax.legend(fontsize=7)
fig.suptitle('Training fit at prespecified checkpoints');fig.text(.5,.015,'Final checkpoint only for held-out inference • Relative1×/4× morphology • Vesuvius Challenge2026, CC BY-NC4.0',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(P/'training-fit.png');plt.close(fig)
calibration=[]
for arm in r['results']:
 support=np.load(P/'training'/f"{arm['held_out']}-support.npy")
 for q in ['native','coarse4']:
  scores=np.load(P/'training'/f"{arm['arm']}-test-{q}.npy")[support];calibration.append(dict(arm=arm['arm'],test_quality=q,predicted_positive_fraction_at_half=float((scores>=.5).mean()),mean_sigmoid=float(scores.mean()),score_quantiles=np.quantile(scores,[0,.01,.5,.99,1]).tolist()))
save(P/'FIXED_THRESHOLD_DIAGNOSTIC.json',dict(scope='Descriptive post-output check; threshold and advancement decision unchanged',rows=calibration));print('Figures and fixed-threshold diagnostic written')
