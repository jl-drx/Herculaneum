"""Render the completed preservation study without model calls or new metrics."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
path=root/'implementation/analysis/metrics.json'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='ff8e549c93ae1430373ff956a5448b5e10abf7202ede858828b0f8192362b751'
m=json.loads(path.read_text())
arms=['frozen_seed42','A_smooth','B_smooth','A_preserve','B_preserve']
names={'frozen_seed42':'Frozen seed42','A_smooth':'Manual, ninth','B_smooth':'Manual + pseudo, ninth','A_preserve':'Manual + preservation','B_preserve':'Manual + pseudo + preservation'}
lines=['''# Tenth study: preservation helps one comparison, but not baseline retention

Adding fixed initial-prediction preservation improves the manual-only continuation on both diagnostic regions, passing that predeclared mitigation screen. It worsens sensitivity with pseudo-labels, and **both new arms still fail retention against frozen seed42**. Keep the frozen baseline. This is a mixed result for a conventional preservation control, not a new model advance or independent ink-accuracy result.

Manual-only mean primary sensitivity rises **73.63% → 76.94%** (+3.31 percentage points); manual-plus-pseudo falls **69.06% → 66.31%** (−2.75 points). Frozen seed42 remains **90.32%**. The shared-mitigation criterion is false. The matched late-batch training losses also remain worse than the initializer, so the result does not establish useful adaptation merely because one retention comparison improves.

## Fixed design and execution

The [protocol](PROTOCOL.md), [lead review](ROOT_REVIEW.md), [independent pre-run review](review/CODE_REVIEW.md) and code were frozen before new model calls. Exactly two new 32-update CPU arms were executed once. Historical ninth A_smooth/B_smooth and frozen seed42 were reused by literal artifact identities. The unchanged initializer predicted the exact same normalized/augmented training tensors as the student, with independent storage, no teacher gradients and unchanged teacher state/RNG. No PHerc0814 data entered gradients or preservation targets.

Original manual/pseudo BCE targets remain 0.25/0.75. A adds mean Bernoulli KL from the initializer on manual participation; B also adds 0.25 times KL on the original pseudo support. Lambda is one, temperature one, with float64 KL arithmetic from FP32 logits. Masks, teacher ancestry, sampling, optimizer, learning rate, update count and calibration remain unchanged. The teacher target is model-derived, not a new annotation. No unknown pixel outside the original active support contributes to the loss.

Ready plan SHA-256: `9e2c9f4f778c3f794e103191fd413937b521fa0d0b72cab0e3905778ce3e3a55`. Metrics SHA: `ff8e549c93ae1430373ff956a5448b5e10abf7202ede858828b0f8192362b751`. There are64 new student updates and64 extra frozen-teacher batch forwards, plus the specified final student evaluation. No extra probe, seed, coefficient search, intermediate checkpoint selection or scientific retry occurred.

## Retention measurements

The two PHerc0814 contexts are reused supplied-supervision development diagnostics from one specimen in a known historical model-development corpus. Equal-context means are descriptive. Calibration uses separate fixed regions and a separately selected threshold for each model; achieved specificity is constrained above95%, not equal across models.

| Model | Mean AUC | Mean AP | Primary sensitivity | Primary specificity |
|---|---:|---:|---:|---:|''']
for a in arms:
 v=m['equal_context_macro'][a]
 lines.append(f"| {names[a]} | {v['auc']:.6f} | {v['ap']:.6f} | {100*v['primary_sensitivity']:.2f}% | {100*v['primary_specificity']:.3f}% |")
lines.append('\n![Preservation comparison](visualization/render/retention-comparison.png)\n')
for case_id,c in m['diagnostic'].items():
 lines.extend([f"### {case_id}\n",f"Source XYWH `{c['bbox_source_xywh']}`; {c['common_primary_count']:,} shared participating pixels, {c['positive']:,} supplied positives and {c['negative']:,} supplied negatives; prevalence {c['prevalence']:.6f}. The legacy ID contains `fresh`; it is now reused development evidence.\n",'| Model | AUC | AP | Primary sensitivity | Specificity | TP | FP | TN | FN |','|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
 for a in arms:
  v=c['models'][a];o=v['operating']['primary'];q=o['confusion']
  lines.append(f"| {names[a]} | {v['auc']:.6f} | {v['ap']:.6f} | {o['sensitivity']:.6f} | {o['specificity']:.6f} | {q['tp']} | {q['fp']} | {q['tn']} | {q['fn']} |")
lines.append('''
Manual preservation improves sensitivity by3.515 and3.105 percentage points relative to ninth, with ranking changes inside the fixed allowances. Against frozen, it gains1.167 points in rank01 but loses27.921 points in rank14, failing baseline retention. The rank01 gain also has lower specificity than frozen; it is not a free dominance improvement.

Pseudo preservation changes sensitivity by−0.020 and−5.475 points relative to ninth, failing the mean-gain and rank14 criteria. Against frozen it loses3.541 and44.472 points, with additional rank14 AUC/AP margin failures. No screen was relaxed or favorable arm silently promoted. Full-precision per-context checks are in the metrics JSON.

### Calibration and interaction

Primary remains the maximum of three per-region conservative empirical95%-specificity cutoffs. Secondary uses PHerc1667/w029 only. Threshold receipts precede diagnostic inference, and historical maps/thresholds remain unchanged.

| Model | Primary cutoff | Secondary cutoff | Secondary mean sensitivity |
|---|---:|---:|---:|''')
for a in arms:
 lock=json.loads(Path(m['thresholds'][a]['path']).read_text())
 lines.append(f"| {names[a]} | {lock['primary']:.12f} | {lock['secondary']:.12f} | {m['equal_context_macro'][a]['secondary_sensitivity']:.6f} |")
lines.append('''
The two new arms have identical primary/secondary cutoffs, both determined by w029. Secondary reporting therefore leaves their result unchanged. Historical B_smooth secondary sensitivity is slightly higher than its primary, making the new B comparison slightly worse under that rule. All per-region secondary confusion counts are retained in the full metrics JSON; for every role except historical B_smooth they equal the primary table above.

| Contrast | ΔAUC | ΔAP | Δprimary sensitivity (pp) | Δprimary specificity (pp) |
|---|---:|---:|---:|---:|''')
for k,v in m['factorial_contrasts'].items():
 lines.append(f"| {k.replace('_',' ')} | {v['auc']:+.6f} | {v['ap']:+.6f} | {100*v['primary_sensitivity']:+.3f} | {100*v['primary_specificity']:+.3f} |")
lines.append('''
The sensitivity difference of differences is−6.058 percentage points. This is a descriptive preservation/pseudo interaction under one continuation schedule, not evidence of a general mechanism. The [retained score audit](diagnostic-audit/DIAGNOSTIC_REPORT.md) separately demonstrates changed ordering, class-specific score redistribution and substantial effects of already fixed cutoffs. These observations cannot locate internal representation damage; diagnostic-specific threshold fitting would not resolve that uncertainty.

## Adaptation cost on exactly matched batches

The table averages preupdate batches25–32 from evolving models. It compares the original target objective, excluding KL, with historical ninth receipts and with the unchanged initializer's actual predictions on those identical inputs. These are online optimization diagnostics, not final-model accuracy. Lower is better; positive deltas versus the initializer mean worse target fit.

| New arm | Manual BCE | Historical manual BCE | Initializer manual BCE | Original objective | Historical objective | Initializer objective |
|---|---:|---:|---:|---:|---:|---:|''')
for a,c in m['adaptation_cost']['arms'].items():
 v=c['last_eight_means']
 lines.append(f"| {names[a]} | {v['loss_U']:.6f} | {v['historical_U']:.6f} | {v['teacher_U']:.6f} | {v['original_objective']:.6f} | {v['historical_original_objective']:.6f} | {v['teacher_original_objective']:.6f} |")
lines.append('''
Manual-only preservation changes original loss by−0.000198 versus ninth but remains+0.005913 above the initializer. Pseudo preservation improves original loss by−0.005043 versus ninth yet remains+0.003658 above the initializer. Both new manual losses are worse than their same-input initializer reference. This is stronger reason to pause loss-variant tuning than the isolated mitigation pass is to scale it.

Late-batch mean manual KL is0.046585/0.045041 for A/B; mean absolute manual probability drift is0.102333/0.088891. B's pseudo KL is0.035316 and pseudo probability drift0.064353. These values measure remaining drift, not a demonstrated reduction against ninth: ninth did not retain its per-step raw logits for that comparison. All current raw logits and per-step curves are retained for reproducibility.

## Independent verification and resources

Both first actual student/teacher outputs match exactly, and first full parameter-gradient norms match their historical counterparts exactly. Initial hard/smoothed losses and normalized input/target/mask hashes agree. Frozen teacher predictions are bitwise identical across the two arms and its semantic state remains unchanged. Each final student checkpoint is a safe raw state dictionary with finite, strictly matching schema/storage aliases and actual parameter changes.

The completed-output independent review reconstructed all64 losses from raw logits and original masks, along with fingerprints, teacher identity, online cost curves, saved checkpoints and calibration/metrics/screens. Its maximum complete-objective discrepancy is1.247e-7, KL discrepancy2.78e-17 and metric discrepancy2.22e-16, within the fixed tolerances. It checked312 unique records and used three safe checkpoint reads plus one factory schema inspection with no forward/backward/inference. See [independent verification](review/result/independent-verification.json). The first reviewer attempt misread the explicit eight-step list as a two-endpoint interval; its failure and isolated checker correction are preserved. No scientific code, result or model run was changed or repeated.

The two workers used30.99/30.13 internal wall seconds,62.62 seconds combined controller time, at most1497.36 MiB peak RSS, and307,780,548 bytes of run/analysis output, below400 MiB. Final checkpoints are138,351,655 bytes each; four raw logit arrays are4,194,432 bytes each. Runtime versions and all prior frozen inputs are unchanged. No new scan/model/label-array acquisition, dependencies, paid compute or automatic scientific retry occurred.

## Reference lineage and next decision

The separate [lineage audit](reference-audit/LINEAGE.md) maps existing adaptation/calibration/diagnostic support and confirms all29 curated representations occur in the four original configurations. It identified **PHerc0009B** and **PHerc0500P2**, absent from those explicit roots, as metadata-only candidates. Eight bounded HTTPS reads totaled36,579 response-body bytes, all successful, with no arrays or predictions inspected. Matching CT/label XY metadata establishes only canvas compatibility. The releases are transferred/model-assisted annotations with unknown original annotation source, incomplete teacher exposure, unresolved label transform/depth relation and a dirty package-commit identifier. They are not independently authenticated ground truth.

Keep both candidate specimens outside current adaptation, teacher-target generation and calibration. PHerc0009B is the first mapping/depth-contract lead;109 planes alone does not justify borrowing the existing pooling range. For independent ink accuracy, an exposed-fragment IR/CT pair with verified alignment and annotation lineage is a stronger path. The catalog also identifies legacy Frag4/PHercParis1Fr39 as an optical-reference provenance lead; its past Kaggle-withheld status does not establish holdout from current models. The reservation proposal is unexecuted, and standing ordinary-research authorization remains in place.

Retain frozen seed42. **Pause further loss/coefficient tuning on this reused panel.** Next complete a defensible fragment reference contract and reserve physical support before new predictions; audit the adaptation task/optimization fit before another training comparison. Do not turn a new crop into an independence claim or add replay/calibration changes to rescue this study. Research, not prize compatibility, drives that decision, while actual source terms and attribution remain binding. No authentic letter, transcription, translation, independent accuracy, significance, novelty or frontier-performance claim follows here.

Paid GPU spend remains£0 and the total first-pilot cap remains£10 including failures/retries and attributable overhead. No R2 write, credential import, publication, submission or outreach occurred. Saved local evidence/startup guidance is not an external backup or proof of fresh-task restoration.
''')
with (root/'RESULTS.md').open('x') as f:f.write('\n'.join(lines).rstrip()+'\n')
print(json.dumps({'path':str(root/'RESULTS.md'),'sha256':hashlib.sha256((root/'RESULTS.md').read_bytes()).hexdigest()}))
