"""Post-run comparisons; never used to select a training endpoint or checkpoint."""
import json,math,os
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def save(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

native=read(P/'TRAINING_RESULTS.json')
fits={r['step']:{c['case']:c['metrics'] for c in r['cases']} for r in native['training_fit']}
growth={c:fits[8192][c]['ap']-fits[4096][c]['ap'] for c in fits[8192]}
fit_pass=all(v['ap']>=.8 for v in fits[8192].values())
budget_status='single_fold_fit_qualified' if fit_pass else 'fit_failed_flat_last_interval' if all(v<.01 for v in growth.values()) else 'fit_failed_still_growing_budget_inconclusive'
panel=read(P/'checkpoint-panel/RESULTS.json');arms={r['checkpoint']['label']:r for r in panel['results']}
def contrast(left,right):
    a,b=arms[left],arms[right];br={r['id']:r for r in b['regions']}
    regional=[dict(id=r['id'],parent=r['parent'],delta={k:r['metrics'][k]-br[r['id']]['metrics'][k] for k in r['metrics']}) for r in a['regions']]
    return dict(left=left,right=right,regions=regional,Paris_equal_parent_delta={k:a['Paris_equal_parent'][k]-b['Paris_equal_parent'][k] for k in a['Paris_equal_parent']},maximum_absolute_regional_AP=max(abs(r['delta']['average_precision']) for r in regional),regions_absolute_AP_above_001=sum(abs(r['delta']['average_precision'])>.01 for r in regional))
contrasts=dict(final_seed_difference=contrast('seed43-75000','seed42-75000'),seed42_late_training=contrast('seed42-75000','seed42-60000'),seed43_late_training=contrast('seed43-75000','seed43-60000'))
save(P/'ANALYSIS.json',dict(native=dict(training_fit= fits,AP_growth_4096_to8192=growth,fit_pass=fit_pass,budget_interpretation=budget_status),checkpoint_contrasts=contrasts,decision='Close profilometry and prioritize PHerc343 positive-control reproduction, then fixed-recipe PHerc0800 transfer. Replace universal CPU AP GPU gate with experiment/data/runtime/cost readiness. No checkpoint promoted by this panel.'))

os.environ.setdefault('MPLCONFIGDIR','/tmp/herculaneum-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/herculaneum-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for case in fits[8192]:
    xs=sorted(fits);axes[0].plot(xs,[fits[s][case]['ap'] for s in xs],marker='o',label=case.split('_')[0])
axes[0].axhline(.8,color='grey',ls='--',label='Prespecified fit criterion')
axes[0].set(xlabel='Optimizer updates',ylabel='Training annotation AP',ylim=(0,1),title='One native profilometry trajectory')
axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
r=contrasts['final_seed_difference']['regions'];xs=np.arange(len(r));delta=[a['delta']['average_precision'] for a in r]
axes[1].barh(xs,delta,color=['#177e89' if d>=0 else '#c14f37' for d in delta]);axes[1].axvline(0,color='black',lw=.8)
for d in [-.01,.01]:axes[1].axvline(d,color='grey',ls='--',lw=1)
axes[1].set_yticks(xs,[a['id'] for a in r],fontsize=8);axes[1].set(xlabel='AP: released seed43 minus seed42, both75k',title='Reused development regions; two seeds')
fig.suptitle('Plinius45: optimization budget and checkpoint variability',fontsize=13)
fig.savefig(P/'overview.png',dpi=150);plt.close(fig)
g=read(P/'geometry/RESULTS.json');fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,seg in zip(axes,sorted(set(r['segment'] for r in g['nodes']))):
    rows=[r for r in g['nodes'] if r['segment']==seg];valid=[r for r in rows if r['measurement']['status']=='measured'];invalid=[r for r in rows if r['measurement']['status']!='measured']
    if invalid:ax.scatter([r['node_yx'][1] for r in invalid],[r['node_yx'][0] for r in invalid],marker='x',color='#aaaaaa',s=18,label='Invalid / tied')
    art=ax.scatter([r['node_yx'][1] for r in valid],[r['node_yx'][0] for r in valid],c=[r['measurement']['offset'] for r in valid],cmap='coolwarm',vmin=-15,vmax=15,s=38)
    ax.invert_yaxis();ax.set(title=seg,xlabel='Mesh node column',ylabel='Mesh node row');ax.set_aspect('equal')
fig.colorbar(art,ax=axes,label='Nearest M7 run offset (native voxels)',shrink=.8)
fig.suptitle('PHerc0800: sparse mesh–M7 disagreement, not surface accuracy')
fig.savefig(P/'geometry-offsets.png',dpi=150);plt.close(fig)
print(json.dumps(dict(native_budget_status=budget_status,training_AP={k:v['ap'] for k,v in fits[8192].items()},native_growth=growth,final_seed_difference=contrasts['final_seed_difference']),indent=2))
