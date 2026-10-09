"""Prespecified descriptive gates, all three parents and four quality combinations."""
import numpy as np
from pixel_http import *
r=json.loads((P/'TRAINING_RESULTS.json').read_text());arms=r['results'];need(len(arms)==6 and all(v['updates']==2048 for v in arms),'Complete fixed budget')
parents=[];fit=[]
for h in range(3):
 n=next(a for a in arms if a['arm']==f'fold{h}-native');c=next(a for a in arms if a['arm']==f'fold{h}-coarse4');get=lambda a,q:next(t['metrics'] for t in a['test'] if t['quality']==q)
 row=dict(held_out=n['held_out'],native_native=get(n,'native'),native_coarse=get(n,'coarse4'),coarse_native=get(c,'native'),coarse_coarse=get(c,'coarse4'),controls=[{k:v for k,v in ctl.items() if k!='fit'} for ctl in n['controls']]);parents.append(row)
 for a in [n,c]:
  for v in a['training_fit'][-1]['cases']:
   previous=next(t for t in a['training_fit'][-2]['cases'] if t['case']==v['case']);fit.append(dict(arm=a['arm'],case=v['case'],ap=v['metrics']['ap'],native_arm=a['train_quality']=='native',ap_change_last_half=v['metrics']['ap']-previous['metrics']['ap'],budget_still_improving=v['metrics']['ap']-previous['metrics']['ap']>=.01))
keys=['native_native','native_coarse','coarse_native','coarse_coarse'];macro={k:{m:float(np.mean([p[k][m] for p in parents])) for m in ['ap','auroc','dice_at_half','brier']} for k in keys};control_macro={q:{name:float(np.mean([next(c['metrics']['ap'] for c in p['controls'] if c['quality']==q and c['name']==name) for p in parents])) for name in ['constant','mean_height','hole_density']} for q in ['native','coarse4']}
native_fit=all(v['ap']>=.8 for v in fit if v['native_arm']);beats_controls=all(macro['native_native']['ap']>=v+.05 for values in control_macro.values() for v in values.values());parent_controls=all(p['native_native']['ap']>=max(c['metrics']['ap'] for c in p['controls']) for p in parents);transfer=native_fit and beats_controls and parent_controls
def minimum_delta(a,b):return min(p[a][m]-p[b][m] for p in parents for m in ['ap','auroc'])
gap=macro['native_native']['ap']-macro['coarse_coarse']['ap'];matched_gain=macro['coarse_coarse']['ap']-macro['native_coarse']['ap'];teacher=transfer and gap>=.01 and minimum_delta('native_native','coarse_coarse')>=-.02;matched=transfer and matched_gain>=.01 and gap<.01 and minimum_delta('coarse_coarse','native_coarse')>=-.02
if not native_fit:decision='NATIVE_POSITIVE_CONTROL_NOT_FIT__RESOLUTION_CONCLUSION_INCONCLUSIVE'
elif not transfer:decision='NATIVE_FIT_WITH_WEAK_TRANSFER__PRIORITIZE_COVERAGE_OR_REPRESENTATION'
elif teacher:decision='NATIVE_ADVANTAGE__LATER_TEACHER_HYPOTHESIS_JUSTIFIED'
elif matched:decision='MATCHED_OBSERVATION_TRAINING_PRIORITY'
else:decision='MIXED_OR_NO_ACTIONABLE_QUALITY_DIFFERENCE'
out=dict(status='COMPLETED_CPU_COMPARISON_NOT_PROMOTED',created_at_utc=now(),decision=decision,parents=parents,macro=macro,control_macro_ap=control_macro,training_fit=fit,gates=dict(native_fit=native_fit,native_macro_controls_margin=beats_controls,no_parent_loses_to_control=parent_controls,native_transfer=transfer,native_matched_quality_AP_gap=gap,matched_training_coarse_AP_gain=matched_gain,native_vs_matched_min_regional_AP_AUROC_delta=minimum_delta('native_native','coarse_coarse'),matched_vs_cross_quality_min_regional_AP_AUROC_delta=minimum_delta('coarse_coarse','native_coarse'),teacher_hypothesis=teacher,matched_quality_priority=matched,native_budget_still_improving=any(v['budget_still_improving'] for v in fit if v['native_arm'])),optimizer_updates=12288,physical_groups=3,pixel_counts_not_independent_samples=True,CT_transfer_established=False,independent_ink_truth=False,model_promoted=False,GPU_spend_GBP=0)
save(P/'LEARNING_ANALYSIS.json',out);print(json.dumps(dict(decision=decision,macro=macro,gates=out['gates']),indent=2))
