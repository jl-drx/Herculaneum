"""Present frozen endpoint measurements without selecting a new model or cutoff."""
import os,json
from common import *
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache');os.environ['XDG_CACHE_HOME']=str(HERE/'plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

m=json.loads((HERE/'METRICS.json').read_text());run=json.loads((HERE/'run-v1/COMPLETE.json').read_text());t=json.loads((HERE/'run-v1/D_replication/COMPLETE.json').read_text())
v=json.loads((HERE/'EVALUATION_VERIFICATION.json').read_text());tv=json.loads((HERE/'TRAINING_VERIFICATION.json').read_text())
roles=['seed42','A_control-512','C_coverage-512','midpoint','D_replication-512']
names=['Seed42','A512','Coverage512 crop1','Fixed midpoint','Replication512 crop2']
colors=['#697787','#d08827','#15776a','#426ab3','#b5334a'];key='average_precision';candidate=roles[-1]
fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for role,name,color in zip(roles,names,colors):
    ax[0].scatter(-m['worst_deltas_vs_seed42'][role][key],m['roles'][role]['equal_parent_Paris'][key],color=color,label=name,s=60)
ax[0].set(xlabel='Worst regional AP loss vs seed42',ylabel='Equal-parent Paris AP');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
ids=list(m['region_deltas']['seed42']);short=[i.replace('pherc0814-46527-fresh-','PHerc0814-').replace('context','c') for i in ids]
for panel,key,title in [(ax[1],'average_precision','Regional AP change vs seed42'),(ax[2],'roc_auc','Regional AUROC change vs seed42')]:
    for role,offset,color in zip(roles[2:],[-.2,0,.2],colors[2:]):panel.scatter([m['roles'][role]['contexts'][i][key]-m['roles']['seed42']['contexts'][i][key] for i in ids],[j+offset for j in range(8)],c=color,s=30)
    panel.set_yticks(range(8),short,fontsize=8);panel.invert_yaxis();panel.axvline(0,color='grey',lw=1);panel.axvline(-.02,color='red',ls=':',lw=1);panel.set_title(title,fontsize=10);panel.grid(axis='x',alpha=.2)
fig.suptitle('Phase40: disjoint crop replication on reused development regions',fontsize=13)
fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)
save(HERE/'FIGURES.json',dict(figure=pin(HERE/'comparison.png'),meaning='Frozen Paris aggregate/worst-region tradeoff plus all regional changes; descriptive, not independent-test performance'))

def fmt(v):return f'{v:.9f}'
rows=[]
for role,name in zip(roles,names):
    r=m['roles'][role]['equal_parent_Paris'];w=m['worst_deltas_vs_seed42'][role]
    rows.append(f"| {name} | {fmt(r['average_precision'])} | {fmt(r['roc_auc'])} | {w['average_precision']:+.9f} | {w['roc_auc']:+.9f} |")
tables=[]
for key,title in [('average_precision','AP'),('roc_auc','AUROC')]:
    lines=[f'### Regional {title} deltas for crop2','', '| Region | vs seed42 | vs A512 | vs crop1 | vs midpoint |','| --- | ---: | ---: | ---: | ---: |']
    for cid in ids:lines.append('| '+cid+' | '+' | '.join(f"{m['region_deltas'][ref][cid][key]:+.9f}" for ref in roles[:-1])+' |')
    tables.append('\n'.join(lines))
risk=[]
for step,row in t['measurements'].items():risk.append((int(step),f"| {step} | {row['mean']['full']:.9f} | {row['mean']['hard_base']:.9f} | {row['mean']['loss_D_M']:.9f} |"))
original=[]
for role,name,path in [('A_control-512','A512',REPORTS/'thirtyseventh-experiment/run-v1/A_control/COMPLETE.json'),('C_coverage-512','Crop1',REPORTS/'thirtyeighth-experiment/run-v1/C_coverage/COMPLETE.json'),('D_replication-512','Crop2',HERE/'run-v1/D_replication/COMPLETE.json')]:
    r=json.loads(path.read_text());v0=r['measurements']['512']['mean'] if role=='A_control-512' else r['original_final']['mean'];original.append(f"| {name} | {v0['full']:.9f} | {v0['hard_base']:.9f} | {v0['loss_D_M']:.9f} |")
screen=m['screen'];relation=m['descriptive_replication'];summary=f"Advancement: **{'PASS' if screen['passes'] else 'FAIL'}**. Descriptive replication of the midpoint tradeoff: **{'PASS' if relation['passes'] else 'FAIL'}**."
text=f'''# Phase40: disjoint source-crop replication

{summary} See [next study recommendation](NEXT_STUDY_DRAFT.md) for the resulting research decision. No GPU resources were used or authorized by these results.

Crop2 has the highest Paris AP of the five fixed states, but its gain over A512 is only 0.000247059. Both mixed runs pass the regional retention floor that A512 breaches. The stricter midpoint relation misses only worst AP retention, by 0.000206010, with the worst region shifting from frag2-context2 (midpoint) to frag3-context1 (crop2). This is a narrow descriptive miss, not evidence that broader training data generally fails. The original-source exposure confound remains.

The frozen 512-update mixed recipe was run once from original seed42, substituting PHerc1667/w013 crop [2560,640,256,256] for crop1 [3968,1408,256,256]. Source rectangles have X/Y gaps of 1152/512 level-2 pixels. This is a spatial replication within one supplied surface coordinate system, not a different surface/scroll, an untouched test or independently verified physical mesh separation.

## Fixed endpoint

| Role | Paris AP | Paris AUROC | Worst regional AP delta | Worst regional AUROC delta |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+f'''

Paris summaries average regions within each parent and weight Paris1/Paris2 equally. Worst deltas are the minima across all eight regions against seed42; PHerc0814 contributes to retention, not Paris aggregation. Crop2 Paris AP gain is {screen['Paris_AP_gains']['seed42']:+.9f} against seed42 and {screen['Paris_AP_gains']['A_control-512']:+.9f} against A512. Advancement requires both >=0.01 and no region below -0.02 in AP or AUROC.

Frozen descriptive components against the single midpoint: Paris AP no lower = {relation['Paris_AP_no_lower_than_midpoint']}; worst AP delta no lower = {relation['worst_AP_delta_no_lower_than_midpoint']}; worst AUROC delta no lower = {relation['worst_AUROC_delta_no_lower_than_midpoint']}. All three are required. This relation does not replace advancement.

Regional advancement failures: `{json.dumps(screen['regional_failures'])}`.

![Aggregate and regional comparisons](comparison.png)

'''+ '\n\n'.join(tables)+'''

Full raw metrics, fixed 0.5/historical-cutoff confusion tables and aggregates for all five roles are in [METRICS.json](METRICS.json). No threshold or coefficient was fitted.

## Optimisation and source exposure

| Update | Fixed mixed objective | Hard-label BCE diagnostic | Seed42 KL |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(row for step,row in sorted(risk))+'''

These risks compare checkpoints on the same fixed crop2 schedule. They are not cross-crop accuracy measures. The following final risks use the identical original PHerc0139 schedule for every arm:

| Model | Original-support objective | Hard-label BCE | Seed42 KL |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(original)+f'''

All arms use the same .25/.75 BCE + unit seed42 KL recipe; crop1 and crop2 replace identical original patch slots. Every mixed update exactly matches A512's total positive/negative counts, with 230,340/387,868 visits per 32-step cycle. Original-source exposure is still half of A512's patch slots. Crop2 uses 46,020 unique additional supervised pixels versus crop1's 55,313, giving total unique training support 100,721 versus 110,014. This is not a pure diversity experiment or a control for unique support size/layout.

## Admission and validation

Stage1's eight predeclared candidates all failed class-count feasibility; no CT chunks or training followed that failure. A separately frozen stage2 expanded only the label inventory and admitted its first candidate. All 32 replacement requirements remained fixed. Total: 68 metadata/label/CT requests, 10,707,142 delivered bytes, four CT chunks. A separate reconstruction verified all eight rejected candidates, the expanded inventory ordering and the accepted candidate. This audit was added after training launch; selection rules and acquisition GOs existed before their corresponding label bodies and all model calls.

Seven pre-run fixtures passed, including exact historical objective/gradient parity across all 32 batches. Independent reconstruction checked the raw CT pooling, central labels, selected supports, crop/augmentation and normalization. All 32 retained PHerc0139 teacher patches matched the historical cache exactly.

One scientific attempt completed: 512 backward/update steps, 32 teacher batch-two forwards, 705 student batch-two forwards and 200 development tile forwards. Controller elapsed time {run['elapsed_seconds']:.3f} seconds; peak training RSS {t['peak_RSS_bytes']:,} bytes. Independent verification checked {tv['loss_rows_checked']} training loss rows / {tv['scalar_values_checked']} scalar values (maximum discrepancy {max(tv['maximum_differences'].values()):.3g}), 80 ranking values (maximum discrepancy {v['maximum_rank_difference']:.3g}), all eight masks and 1,179,648 score pixels / 200 sigmoid-Hann blended tiles (maximum discrepancy {v['maximum_blend_difference']:.3g}). It recomputes retained scalar arithmetic, not autograd/optimizer trajectories. All aggregates, contrasts and decisions were independently checked. The prelaunch AST correction is recorded separately; no scientific retry occurred.

All evaluation contexts remain reused development. Supplied labels have incomplete human/teacher ancestry. Frag6 optical registration and physical depth remain unresolved; all original 17 input planes are retained. Reserved parents were untouched. No independent ink accuracy, readable text, transcription or translation has been established. The updated local R2 handoff still reports no Runpod secret import or worker injection; this study does not rely on it. CPU monetary spend is unrestricted, and the £10 total first GPU pilot cap remains in force with GPU work deferred.
'''
with (HERE/'RESULTS.md').open('x') as f:f.write(text)
print(json.dumps(dict(status='REPORT_WRITTEN',advancement=screen['passes'],descriptive=relation['passes'])))
