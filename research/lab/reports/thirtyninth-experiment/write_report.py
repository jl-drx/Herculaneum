"""Render the completed fixed comparison; no new scientific choices or scoring."""
import json
from pathlib import Path
import os
from common import HERE, KEYS, ROLES, pin, save

os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

m=json.loads((HERE/'run-v1/METRICS.json').read_text())
v=json.loads((HERE/'run-v1/VERIFICATION.json').read_text())
labels={'seed42':'Seed42','A_control-512':'A512','C_coverage-512':'Coverage512','midpoint':'Fixed midpoint'}
colors={'seed42':'#697787','A_control-512':'#d08827','C_coverage-512':'#15776a','midpoint':'#426ab3'}
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for role in ROLES:
    axes[0].scatter(-m['worst_deltas_vs_seed42'][role]['average_precision'],m['roles'][role]['equal_parent_Paris']['average_precision'],color=colors[role],s=65,label=labels[role])
axes[0].set(xlabel='Worst regional AP loss vs seed42 (smaller is better)',ylabel='Equal-parent Paris AP (larger is better)')
axes[0].legend(loc='lower right',fontsize=8);axes[0].grid(alpha=.2)
ids=list(m['region_deltas_vs_seed42']['midpoint'])
short=[i.replace('pherc0814-46527-fresh-','PHerc0814-').replace('context','c') for i in ids]
for ax,key,title in [(axes[1],'average_precision','Regional AP change vs seed42'),(axes[2],'roc_auc','Regional AUROC change vs seed42')]:
    for role,offset in [('A_control-512',-.2),('C_coverage-512',0),('midpoint',.2)]:
        ax.scatter([m['region_deltas_vs_seed42'][role][i][key] for i in ids],[j+offset for j in range(8)],color=colors[role],s=27,label=labels[role])
    ax.axvline(0,color='grey',lw=1);ax.axvline(-.02,color='red',ls=':',lw=1)
    ax.set_yticks(range(8),short,fontsize=8);ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2)
fig.suptitle('Phase39: one fixed midpoint on eight reused development regions',fontsize=13)
target=HERE/'comparison.png';fig.savefig(target,dpi=160);plt.close(fig)
save(HERE/'FIGURES.json',dict(figure=pin(target),description='Descriptive aggregate/retention tradeoff and all eight regional AP/AUROC deltas; no uncertainty or independence claim'))

rows=[]
for role in ROLES:
    r=m['roles'][role]['equal_parent_Paris'];w=m['worst_deltas_vs_seed42'][role]
    rows.append(f"| {labels[role]} | {r['average_precision']:.9f} | {r['roc_auc']:.9f} | {w['average_precision']:+.9f} | {w['roc_auc']:+.9f} |")
regional=[]
for cid in ids:
    a=m['region_deltas_vs_seed42']['midpoint'][cid];c=m['region_deltas_vs_seed42']['C_coverage-512'][cid]
    regional.append(f"| {cid} | {a['average_precision']:+.9f} | {c['average_precision']:+.9f} | {a['roc_auc']:+.9f} | {c['roc_auc']:+.9f} |")
text='''# Phase39 results: fixed midpoint does not match the coverage tradeoff

The fixed midpoint passes the eight-region retention limit but fails the AP advancement gate. Coverage512 has higher equal-parent Paris AP and smaller worst AP and AUROC losses than this midpoint. Retain seed42 and defer GPU work. The next useful trained study is a single preselected disjoint PHerc1667 region replication under the same matched total class-exposure design; no such run is included here.

## Frozen comparison and results

Exactly one midpoint was formed from retained FP32 seed42 and A512 score maps using float64 arithmetic: `0.5*seed42 + 0.5*A512`. The coefficient was proposed before this analysis; GO froze 88 input/code pins before its outputs. This is a post-result comparison relative to phase38 and a prediction comparator, not a trained checkpoint or proof of equivalence to fewer updates. No new model execution or training occurred.

| Role | Paris AP | Paris AUROC | Worst regional AP delta | Worst regional AUROC delta |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+'''

Worst deltas are the minima across all eight regions against seed42. Paris summaries give equal weight to the two Paris parents, averaging regions within each parent; PHerc0814 contributes to retention, not that aggregate.

Midpoint AP increases by 0.002762383 against seed42 and is 0.002197490 below A512. Both gains fail the frozen +0.01 requirement; no midpoint region breaches the -0.02 AP/AUROC floor. The separate descriptive relation fails all three components: midpoint AP is 0.001050862 below coverage512, its worst AP delta is 0.007173757 lower, and its worst AUROC delta is 0.005413945 lower.

## All regional comparisons

Every entry below is a delta against seed42. Full raw metrics for all four roles, all regional deltas, fixed-threshold diagnostics and reference reproduction are in [METRICS.json](run-v1/METRICS.json).

| Region | Midpoint AP delta | Coverage AP delta | Midpoint AUROC delta | Coverage AUROC delta |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(regional)+'''

Coverage512 is not uniformly better by region: for example, midpoint retains a larger AP improvement on frag3-context2. Its advantage here is the specifically frozen aggregate/worst-region tradeoff. No uncertainty interval or population significance is claimed.

![Comparison](comparison.png)

## Verification and interpretation

Four contract fixtures passed, including exact-tie ranking and the half-ULP precision case. The separate verifier reconstructed all eight canonical masks, all 1,179,648 midpoint pixels exactly, 64 AP/AUROC values (maximum discrepancy 1.11e-16), parent aggregates, all reported deltas and both decisions. The three prior references reproduced their retained ranking metrics and threshold counts. One scientific attempt completed; zero retries, neural forwards, optimizer updates, acquisitions or GPU use.

This rejects only one fixed reduction of prediction movement as an explanation sufficient to reproduce this particular coverage tradeoff. It does not rule out other coefficients, fewer updates or stronger preservation objectives; none were searched. The data-diversity mechanism remains confounded because phase38 replaced half the original PHerc0139 patch exposures while matching total positive/negative counts. A disjoint source-region replication would test crop dependence, not remove that confound.

All eight evaluation regions are reused development. Supplied manual-plus-iterative references have incomplete ancestry and are not authenticated independent ink truth. No optical alignment, physical depth, new readable text, transcription or translation has been established. Frag6 registration/depth remain unresolved; full original 17-plane predictions and all reserved parents remain unchanged. CPU cost is not constrained by the user; the £10 total first GPU pilot cap remains and GPU work is deferred.
'''
with (HERE/'RESULTS.md').open('x') as stream:stream.write(text)
