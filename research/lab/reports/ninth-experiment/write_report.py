"""Render descriptive tables from frozen ninth outputs, without model calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
metrics_path = ROOT / 'implementation/analysis/metrics.json'
assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == '5b70146072474d00998bd82dc164c0cf0e6cedf4a570418dedaeb5066f11cad0'
metrics = json.loads(metrics_path.read_text())
arms = ['frozen_seed42', 'A', 'B', 'A_smooth', 'B_smooth']
names = dict(frozen_seed42='Frozen seed42', A='A0 manual', B='B0 manual + pseudo', A_smooth='A0.5 manual', B_smooth='B0.5 manual + pseudo')
lines = ["""# Ninth study: smoothing does not consistently preserve retention

The two fixed CPU continuations completed successfully, but both failed the predeclared mitigation and baseline-retention screens. Restoring the published BCE target convention improved equal-context mean AUC/AP for both comparisons; sensitivity transferred inconsistently. Keep the unchanged seed42 baseline. These are descriptive supplied-annotation retention diagnostics on two reused PHerc0814 contexts, not independent ink accuracy or a reading.

Manual-only smoothing changed mean primary sensitivity from **68.89% to 73.63%** (+4.74 percentage points), but reduced it by 5.22 points in rank14. With pseudo-labels, smoothing reduced mean sensitivity from **77.13% to 69.06%** (−8.07 points). Frozen seed42 remains **90.32%**. All models meet the 95% specificity floor in both contexts; the floor is a constraint, not equality of their achieved operating points.

## Fixed comparison

Only A0.5 and B0.5 were trained: 32 updates each, **64 new updates total**, from the same seed42 state. The completed eighth A0/B0 runs and frozen baseline were reused by their exact artifact identities. The failed balanced-anchor C arm was not continued. A uses mean manual BCE; B adds 0.25 times mean fixed hard-teacher BCE. Targets inside the same masks change from 0/1 to 0.25/0.75. Input bytes, teacher, ordered patch/D4 schedule, optimizer, learning rate, masks, update count and calibration rules remain fixed.

The [protocol](PROTOCOL.md), [lead review](ROOT_REVIEW.md), [independent code review](review/CODE_REVIEW.md) and [ready plan](implementation/final-plan/control-plan.json) were fixed before new execution. Plan SHA-256: `9d72d3177d5aa68cd5d59db6f316ed1a5c335f442b6c5c2439f45bfdaebe29cd`. Source audit confirms the BCE formula, while the published original objective also includes Dice. This control restores only the declared BCE convention, not the full historical training recipe.

## Measured outcomes

Equal-context means give each context equal weight. They are two contexts from one specimen, not independent replicates. AUC/AP concern agreement with supplied annotations. Primary thresholds are selected separately for each model using only the same three calibration regions; no diagnostic labels determine thresholds.

| Model | Mean AUC | Mean AP | Primary sensitivity | Primary specificity |
|---|---:|---:|---:|---:|"""]
for arm in arms:
    value = metrics['equal_context_macro'][arm]
    lines.append(f"| {names[arm]} | {value['auc']:.6f} | {value['ap']:.6f} | {100*value['primary_sensitivity']:.2f}% | {100*value['primary_specificity']:.3f}% |")
lines.append('\n![Retention diagnostic comparison](visualization/render/retention-comparison.png)\n')
for case_id, case in metrics['diagnostic'].items():
    lines.extend([f"### {case_id}\n", f"Source bbox xywh `{case['bbox_source_xywh']}`; {case['common_primary_count']:,} participating pixels, {case['positive']:,} supplied positives and {case['negative']:,} supplied negatives; prevalence {case['prevalence']:.6f}. All five models share the complete fixed primary domain. The legacy filename retains `fresh`; its current role is **reused development retention diagnostic**.\n",
        '| Model | AUC | AP | Primary sensitivity | Specificity | TP | FP | TN | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for arm in arms:
        value=case['models'][arm]; op=value['operating']['primary']; c=op['confusion']
        lines.append(f"| {names[arm]} | {value['auc']:.6f} | {value['ap']:.6f} | {op['sensitivity']:.6f} | {op['specificity']:.6f} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} |")
lines.append("""
## Calibration and secondary rule

Primary is the maximum of three per-region empirical 95%-specificity cutoffs with conservative ties. Secondary uses PHerc1667/w029 alone. New threshold files were written before diagnostic inference. Literal historical thresholds and maps are reused. Raw sigmoid scores are not calibrated ink probabilities.

| Model | Primary cutoff | Secondary cutoff | Primary-determining region |
|---|---:|---:|---|""")
for arm in arms:
    lock=json.loads(Path(metrics['thresholds'][arm]['path']).read_text())
    determining=max(lock['development'],key=lambda key:lock['development'][key]['selection']['threshold'])
    lines.append(f"| {names[arm]} | {lock['primary']:.12f} | {lock['secondary']:.12f} | {determining} |")
lines.append("""
The secondary rule is identical to primary for frozen/A0/B0/A0.5. Only B0.5 differs:

| Context | Secondary sensitivity | Specificity | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|""")
for case_id,case in metrics['diagnostic'].items():
    op=case['models']['B_smooth']['operating']['secondary'];c=op['confusion']
    lines.append(f"| {case_id} | {op['sensitivity']:.6f} | {op['specificity']:.6f} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} |")
lines.append("""
B0.5 mean sensitivity becomes 69.56% under the secondary rule, still below B0's 77.13% and frozen's 90.32%. Thus the small change in which region determines the maximum cutoff does not reverse the result. Ranking improvements with weaker sensitivity indicate a score-scale/threshold-transfer issue as well as incomplete retention; they do not by themselves isolate its cause.

## Paired effects and fixed screens

The table reports equal-context continuous effects. Sensitivity/specificity differences are percentage points; AUC/AP differences are raw metric units.

| Contrast | ΔAUC | ΔAP | Δprimary sensitivity (pp) | Δprimary specificity (pp) |
|---|---:|---:|---:|---:|""")
for contrast,value in metrics['factorial_contrasts'].items():
    lines.append(f"| {contrast.replace('_',' ')} | {value['auc']:+.6f} | {value['ap']:+.6f} | {100*value['primary_sensitivity']:+.3f} | {100*value['primary_specificity']:+.3f} |")
lines.append("""
Both mitigation screens fail. A0.5 passes the mean-gain requirement but loses 5.223 percentage points of sensitivity in rank14, beyond the predeclared 2-point allowance. B0.5 loses 2.072 and 14.076 points in the two contexts and fails the mean-gain requirement. Mean AUC/AP improvements are driven by rank14; both ranking measures decline slightly in rank01 for both smoothed pairs.

Both baseline-retention screens also fail. A0.5 sensitivity is 2.348/31.026 points below frozen; B0.5 is 3.520/38.997 points below. A0.5 also exceeds the rank01 AUC-loss allowance and the rank14 AP-loss allowance; B0.5 exceeds the rank14 AP-loss allowance. Comparisons use full precision: A0.5's rank14 AP delta is −0.01000266036, just beyond −0.01, but the large sensitivity loss independently fails retention. No margin was relaxed after inspection.

The sensitivity difference of differences is −12.817 percentage points. This describes an interaction between our smoothing and pseudo-label factors under this fixed continuation/calibration design. One seed and two reused contexts cannot establish a general causal mechanism. The hypothesis that smoothing alone consistently mitigates the earlier retention loss is unsupported by this screen.

## Execution and independent verification

The first actual forward reproduced historical hard-target U=0.31660833954811096 and P=0.3718268871307373 exactly in both arms. Their independently reconstructed normalized input/mask/target hash matches `571d10c58008cb6f9e581a717bbb067846dae6a2ee2d29bdf9eb141ca9a02ec3`. Double-precision softplus checks confirm the smoothed U/P within the fixed tolerance. No extra forward/update was added for this sentinel. Artificial-logit fixtures independently checked smoothing derivatives, zero-smoothing equivalence and zero gradients outside participation.

Independent completed-output verification passed: 252 stable artifact receipts, maximum numeric metric difference 2.22×10⁻¹⁶, exact objective coefficients, ordered exposures and 64 actual new update receipts. Both saved states strictly match schema/aliases, load with `weights_only=True`, and have all 264 canonical parameter tensors changed from the initializer. This audits receipts, tensors and metrics without replaying training or performing extra model inference. See [independent verification](review/result/independent-verification.json).

Workers used 25.47/25.67 seconds internally, 52.47 seconds combined controller wall time, and at most 1,371.22 MiB peak RSS. Run plus analysis outputs total **289,969,467 bytes**, below 400 MiB. Two final checkpoints are 138,351,655 bytes each; no intermediate checkpoint selection or retry occurred. New source verification used two public-code reads totaling 21,098 response-body bytes; no new scan/model download occurred. Model work was offline.

CPU runtime is Python 3.12.14, torch 2.12.0+cpu, NumPy 2.5.3, four threads. Recorded historical torch/thread settings match; full historical package/hardware equality is not established. Real first-step agreement and pinned artifacts strengthen the comparison without making a cross-environment bitwise claim.

## Decision and limits

Retain frozen seed42; do not promote either new checkpoint or scale these recipes onto a GPU. Keep the complete negative result and mixed ranking effects. The next research step is to separate continuation stability from calibration transfer and strengthen reference lineage/physical reservation before further model claims. A post-result [next-study draft](NEXT_STUDY_DRAFT.md) describes that work; it is not a frozen or executed tenth experiment.

The inspected contexts are in a known historical model-development corpus, have unresolved original pixel/teacher exposure and supplied-annotation lineage, and use a pooled-fine representation. They establish no independent ink accuracy, generalization, authenticated letters, transcription, translation or novelty. Research is primary; licensing/access/source-attribution obligations still apply, and prize eligibility has not been asserted.

Paid GPU spend remains **£0**; the aggregate first-pilot limit remains **£10** including failures/retries and attributable overhead. No R2 upload, publication, prize submission or credential import occurred. Local integrity and saved startup guidance are separate from backup/publication/restoration. Full precision metrics, confusion counts, score quantiles, raw maps and curves remain in [the analysis](implementation/analysis/metrics.json).
""")
with (ROOT/'RESULTS.md').open('x') as stream:
    stream.write('\n'.join(lines).rstrip()+'\n')
print(json.dumps(dict(report=str(ROOT/'RESULTS.md'),sha256=hashlib.sha256((ROOT/'RESULTS.md').read_bytes()).hexdigest())))
