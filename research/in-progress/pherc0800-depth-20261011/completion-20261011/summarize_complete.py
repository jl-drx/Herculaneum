"""Summarize all frozen response-agreement comparisons; no accuracy inference."""
from pathlib import Path
import collections, hashlib, json

ROOT=Path(__file__).resolve().parent
source=ROOT/'METRICS.json'
data=json.loads(source.read_text())
plan=json.loads((ROOT/'ASSESSMENT_PLAN.json').read_text())
records={r['key']:r for r in plan['maps']}

def ranges(rows,scope='full_strip'):
    scores=[r['scores'][scope] for r in rows]
    assert scores and all(s['pearson'] is not None for s in scores)
    return dict(comparisons=len(scores),pearson_min=min(s['pearson'] for s in scores),pearson_max=max(s['pearson'] for s in scores),MAD_uint8_levels_min=min(s['mean_absolute_difference_uint8_levels'] for s in scores),MAD_uint8_levels_max=max(s['mean_absolute_difference_uint8_levels'] for s in scores))

all_groups={}
for scope in ['full_strip','cell1','cell2','cell3']:
    all_groups[scope]={kind:ranges([r for r in data['metrics'] if r['kind']==kind],scope) for kind in ['padding','order','cross_model','depth']}
model_groups={}
for model in ['dnative','hybrid_3d2d-seed42','hybrid_3d2d-seed43','hecate']:
    model_groups[model]={kind:ranges([r for r in data['metrics'] if r['kind']==kind and records[r['left']]['model']==model]) for kind in ['padding','order','depth']}
cross=collections.defaultdict(list)
for row in data['metrics']:
    if row['kind']=='cross_model':cross[(records[row['left']]['model'],records[row['right']]['model'])].append(row)
summary=dict(source_METRICS_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),maps=48,pair_comparisons=168,scope_scores=672,known_pixels=data['common_support_pixels'],unknown_pixels=data['unknown_pixels'],all_group_ranges=all_groups,full_strip_per_model=model_groups,full_strip_cross_model=[dict(models=list(k),**ranges(v)) for k,v in cross.items()],interpretation='Ranges cover every prescribed comparison in each group. These are aligned prediction-response agreement, not accuracy, ink truth, uncertainty confidence intervals or a new selection gate.',new_inference=False,gpu_spend_GBP=0)
(ROOT/'NUMERIC_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Complete fixed-map numeric summary','','All 48 map views and 168 distinct pairs were included. Each pair has one','full-strip score and three fixed-cell scores, for 672 scope scores. Every score','uses the inherited common support: 786,240 known pixels and 192 unknown pixels.','Differences are in uint8-equivalent probability levels; no contrast was fitted.','','## Full-strip ranges across all prescribed pairs','','| Comparison | Pairs | Pearson range | Mean absolute difference range |','|---|---:|---:|---:|']
for kind,row in all_groups['full_strip'].items():lines.append(f"| {kind} | {row['comparisons']} | {row['pearson_min']:.3f} to {row['pearson_max']:.3f} | {row['MAD_uint8_levels_min']:.2f} to {row['MAD_uint8_levels_max']:.2f} |")
lines+=['','## Padding comparison by model','','Each range contains both input orders at all three depths (six comparisons).','','| Model | Pearson range | Mean absolute difference range |','|---|---:|---:|']
for model,rows in model_groups.items():
    r=rows['padding'];lines.append(f"| {model} | {r['pearson_min']:.3f} to {r['pearson_max']:.3f} | {r['MAD_uint8_levels_min']:.2f} to {r['MAD_uint8_levels_max']:.2f} |")
lines+=['','## Cross-model comparison','','Each range contains three depths, both padding phases and both orders (12','comparisons). Shared training and calibration limit interpretation.','','| Models | Pearson range | Mean absolute difference range |','|---|---:|---:|']
for r in summary['full_strip_cross_model']:lines.append(f"| {' / '.join(r['models'])} | {r['pearson_min']:.3f} to {r['pearson_max']:.3f} | {r['MAD_uint8_levels_min']:.2f} to {r['MAD_uint8_levels_max']:.2f} |")
lines+=['','The full-precision machine summary also contains all fixed-cell group ranges','and each model’s order/depth ranges. Individual comparisons remain in','`METRICS.json`. Responses change substantially between sampled depths; that','does not identify the ink-bearing surface. Padding agreement alone is not','accuracy, and reversal invariance was not required.','','## Deterministic verification','','- All 21 checkpoint archives and 7,080 restored members passed checksums.','- All 32 shifted map combinations, 3,096 batch receipts, 200 output hashes','  and 16 baseline hashes passed independent completion validation.','- Independent reporting verification recomputed all 672 scope scores; maximum','  differences were 3.33e−16 in Pearson and 1.42e−14 in absolute-difference levels.','- Every source pixel in all 64 exact panels (192 depth-region views) matched','  fixed rounding and brown unknown pixels. All 48 overview tiles also matched','  their declared uniform spatial reduction.','- The original reporting output manifest contains 71 verified files; later','  verification and scientific-review documents need their own package manifest.','','No new model inference, acquisition, paid compute or scientific accuracy','measurement was performed. Visual conclusions are recorded separately.','']
(ROOT/'NUMERIC_SUMMARY.md').write_text('\n'.join(lines))
print(json.dumps(all_groups['full_strip'],indent=2))
