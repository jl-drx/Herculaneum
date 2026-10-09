"""Report fixed results without further model execution or selection."""
import os
from common import *
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache');os.environ['XDG_CACHE_HOME']=str(HERE/'plot-cache/xdg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=json.loads((HERE/'METRICS.json').read_text());p=plan();r=json.loads((HERE/'run-v1/E_no_new_BCE/COMPLETE.json').read_text());d=json.loads(verify(p['full_training_baseline']).read_text());run=json.loads((HERE/'run-v1/COMPLETE.json').read_text());tv=json.loads((HERE/'TRAINING_VERIFICATION.json').read_text());ev=json.loads((HERE/'EVALUATION_VERIFICATION.json').read_text())
roles=['seed42','A_control-512','C_coverage-512','midpoint','D_replication-512','E_no_new_BCE-512'];role=roles[-1];decision=m['supervision_contribution'];screen=m['screen']
names=['Seed42','Original source','Crop1 mixture','Fixed midpoint','Crop2 full loss','Crop2 without new BCE']
fig,ax=plt.subplots(1,2,figsize=(12,4.8));colors=['#8a959e']*4+['#286f8e','#ab593b']
values=[m['roles'][x]['equal_parent_Paris']['average_precision'] for x in roles]
ax[0].barh(names,values,color=colors);ax[0].set_xlim(.30,.325);ax[0].invert_yaxis();ax[0].set_title('Equal-parent Paris average precision')
for i,v in enumerate(values):ax[0].text(v+.00025,i,f'{v:.5f}',va='center',fontsize=9)
steps=[0,*UPDATES]
ax[1].plot(steps,[d['measurements'][str(s)]['mean']['full'] for s in steps],'-o',label='Full mixture',color=colors[-2])
ax[1].plot(steps,[r['measurements'][str(s)]['mean']['full'] for s in steps],'-o',label='No new BCE gradient',color=colors[-1])
ax[1].set_title('Same full objective on fixed mixed support');ax[1].set_xlabel('Optimizer updates');ax[1].legend()
for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='x' if a is ax[0] else 'y',alpha=.2);a.set_axisbelow(True)
fig.suptitle('One fixed supervision-gradient control · 512 CPU updates',fontsize=14)
fig.text(.02,.01,'All eight evaluation regions are reused development. Training loss and reference agreement do not establish ink recovery.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(HERE/'comparison.png',dpi=150);plt.close(fig)
save(HERE/'FIGURES.json',dict(figure=pin(HERE/'comparison.png'),source=pin(HERE/'METRICS.json'),scope='Fixed endpoint and identical full-objective training support'))
table=[]
for x in roles:
 a=m['roles'][x]['equal_parent_Paris'];w=m['worst_deltas_vs_seed42'][x]
 table.append(f"| {x} | {a['average_precision']:.9f} | {a['roc_auc']:.9f} | {w['average_precision']:+.9f} | {w['roc_auc']:+.9f} |")
risk=[]
for s in steps:
 a=r['measurements'][str(s)]['mean'];b=d['measurements'][str(s)]['mean']
 risk.append(f"| {s} | {b['full']:.9f} | {a['full']:.9f} | {a['optimized']:.9f} | {a['loss_U_original']:.9f} | {a['loss_U_new']:.9f} | {a['loss_D_M']:.9f} |")
regional=[]
for cid,row in m['region_deltas']['seed42'].items():
 other=m['region_deltas']['D_replication-512'][cid]
 regional.append(f"| {cid} | {row['average_precision']:+.9f} | {row['roc_auc']:+.9f} | {other['average_precision']:+.9f} | {other['roc_auc']:+.9f} |")
original=[]
for label,item in [('D512',d),('E512',r)]:
 a=item['original_final']['mean'];original.append(f"| {label} | {a['full']:.9f} | {a['hard_base']:.9f} | {a['loss_D_M']:.9f} |")
text=f'''# Experiment41: contribution of the new crop's supervised loss

One frozen 512-update CPU control completed. **Advancement: {screen['passes']}. Prespecified descriptive mechanism result: `{decision['outcome']}`.** The control keeps phase40's exact inputs, label-selected support and teacher KL, while removing only PHerc1667 crop2's BCE gradient. It is not a label-free control and does not remove the new crop's preservation term.

The new crop's supervision improves aggregate Paris AP by 0.001556856 and worst AUROC retention by 0.008139835 relative to this control, while worsening worst AP retention by 0.006896297. Neither arm dominates all three prespecified quantities. Both pass the regional floors and fail the material-gain screen. Keep seed42 as the baseline and move the main research effort to acquisition learnability; no GPU work is justified by this result.

## Fixed endpoint

| Role | Paris AP | Paris AUROC | Worst AP delta vs seed42 | Worst AUROC delta vs seed42 |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(table)+f'''

Control AP gains: {screen['Paris_AP_gains']['seed42']:+.9f} vs seed42 and {screen['Paris_AP_gains']['A_control-512']:+.9f} vs A512. Advancement requires both >=.01 and no region AP/AUROC drop below-.02. Regional failures: `{json.dumps(screen['regional_failures'])}`.

The control-minus-full-mixture contrasts are `{json.dumps(decision['control_minus_full_mixture'])}` for Paris AP, worst regional AP retention and worst regional AUROC retention. The prespecified rule compares all three: control no worse on all means new-crop BCE unnecessary for this observed tradeoff; full mixture no worse on all and strictly better on one supports its supervised-gradient contribution; otherwise the result is a tradeoff. This is descriptive, without a significance test or new independent evaluation sample.

![Fixed result and common training objective](comparison.png)

| Region | AP delta vs seed42 | AUROC delta vs seed42 | AP delta vs full mixture | AUROC delta vs full mixture |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(regional)+'''

Paris aggregates weight Paris1 and Paris2 equally after averaging regions within each parent. PHerc0814 enters the regional retention screen, not the Paris aggregate. Full raw metrics, threshold diagnostics and reference contrasts are retained in METRICS.json.

## Fitting the unchanged objective

| Update | Full-mixture full objective | Control full objective | Control optimized objective | Original BCE/N | Removed new BCE/N | Joint KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(risk)+'''

The original full objective is evaluated on identical mixed batches for both runs. The reduced optimized objective omits a positive term and cannot be compared numerically to the full objective as evidence of better fitting. Source contributions use the original joint supported denominator, not per-patch renormalization. Original-only final diagnostics use the same complete PHerc0139 schedule:

| Model | Original-support full objective | Hard-label BCE | Seed42 KL |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(original)+f'''

## Validation and interpretation limits

Eight fixtures passed before GO, including all 32-batch full-objective parity, supervised-gradient decomposition, bitwise preservation of original-slot gradients, retained new-slot KL, fixed-support new-label invariance and exact input/teacher identity. Two preflight assertions initially failed on FP32 reduction-order and tiny KL cancellation differences; the correction and original failed receipts are retained in PREFLIGHT_NOTES.md/preflight-v1. No scientific model call preceded GO.

GO SHA256 `{pin(HERE/'GO.json')['sha256']}` froze 538 files. One scientific attempt, no retries: 512 backwards/updates, 705 student batch-two forwards, 200 development tile forwards and zero new teacher forwards. The 32 teacher batches were reused byte-for-byte. Controller time {run['elapsed_seconds']:.3f}s; peak training RSS {r['peak_RSS_bytes']:,} bytes. No new scientific data acquisition or GPU/R2 activity.

Independent numerical verification reconstructed {tv['loss_rows_checked']} loss rows / {tv['scalar_values_checked']} scalar values (maximum difference {max(tv['maximum_differences'].values()):.3g}), {ev['rank_values_checked']} ranking values (maximum {ev['maximum_rank_difference']:.3g}), all eight canonical masks and {ev['blended_score_pixels']:,} score pixels / 200 raw tiles (maximum blend difference {ev['maximum_blend_difference']:.3g}). It checks saved arithmetic, not a replay of autograd or optimizer trajectories. Measurement-isolation and optimizer-continuity receipts are retained.

The intervention identifies the effect of this supervised-gradient term within one deterministic design, conditional on annotation-informed support, teacher preservation and shared optimizer. It cannot establish a pure diversity benefit, isolate all reduced-PHerc0139-exposure effects or provide an independent generalization estimate. All eight regions remain reused development. Supplied label ancestry and optical/physical-depth questions remain unresolved; all 17 input planes were retained. No authenticated ink, transcription or translation is claimed.

## Research direction

The paired-acquisition agenda is assessed in [PAIRED_ACQUISITION_REVIEW.md](PAIRED_ACQUISITION_REVIEW.md). A bounded metadata audit found public Paris4 old/new volumes, directed transforms and paired meshes. Matrix consistency passed, but local landmark accuracy and an independent reference subset remain unverified. No paired image or mesh-coordinate payload was fetched.

Prioritize the [proposed observation-quality comparison](NEXT_STUDY_DRAFT.md): native, synthetic coarse and registered real coarse observations, with matched training and cross-quality testing on separate physical blocks. This proposal has not run. It should precede privileged-observation distillation and any conditional geometry model. CPU monetary spend is unrestricted; GPU work remains deferred under the £10 total first-pilot cap. Runpod secret import and boot-time delivery remain unverified.
'''
with (HERE/'RESULTS.md').open('x') as f:f.write(text)
print(json.dumps(dict(status='REPORT_WRITTEN',screen=screen,mechanism=decision)))
