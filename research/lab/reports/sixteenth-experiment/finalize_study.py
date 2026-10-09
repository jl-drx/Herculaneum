"""Bind completed, independently checked outputs and write additive research records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent
EXPECTED_METRICS = 'c911803a07ac44741eb9724b2e4b78dbfd8cb32f92848378960960f17d1ee175'


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def read(relative):
    return json.loads((HERE / relative).read_text())


def write_new(path, value):
    path = Path(path)
    assert not path.exists(), 'Keep previous records immutable: ' + str(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def records_in(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        for child in value.values():
            yield from records_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from records_in(child)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--independent-verification', type=Path, required=True)
parser.add_argument('--presentation-receipt', type=Path, required=True)
args = parser.parse_args()
metrics_pin = record(HERE / 'inference/analysis/metrics.json')
assert metrics_pin['sha256'] == EXPECTED_METRICS
metrics = read('inference/analysis/metrics.json')
independent_pin = record(args.independent_verification)
independent = json.loads(args.independent_verification.read_text())
assert independent['status'] == 'PASS'
assert any(r['path'] == metrics_pin['path'] and r['sha256'] == EXPECTED_METRICS
           for r in records_in(independent)), 'Independent PASS must bind exact metrics'
presentation_pin = record(args.presentation_receipt)
presentation = json.loads(args.presentation_receipt.read_text())
assert presentation['status'] == 'PASS' and presentation['metrics'] == metrics_pin
ct = read('numeric-review/production-verification/completed-review/verification.json')
assert ct['status'] == 'PASS' and ct['all_CT_arrays_exact'] is True
run = read('inference/run/results.json')
metadata = read('production-acquisition/metadata-run/result.json')
pixels = read('production-acquisition/pixel-run/result.json')
pooled = read('production-pooling-v2/pooling-run/result.json')
assert all(v['status'] == 'PASS' for v in (run, metadata, pixels, pooled))
assert run['tile_forward_count'] == 450
assert run['model_load_operations']['total_load_state_dict_calls'] == 10
assert metrics['primary_specificity_transfer_all_four'] == {
    'A_preserve': False, 'B_preserve': False, 'frozen_seed42': False}

roles = ['frozen_seed42', 'A_preserve', 'B_preserve']
names = {'frozen_seed42': 'Seed42 baseline', 'A_preserve': 'A preserve', 'B_preserve': 'B preserve'}
table = ['| Fixed model | Mean ROC AUC | Mean AP | Sensitivity, old → Paris1 cutoff | Specificity at Paris1 cutoff | Regions passing ≥95% specificity |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for role in roles:
    value = metrics['paris2_equal_fragment'][role]
    old = value['operating']['retained_fourteenth']
    new = value['operating']['frag3_only']
    passed = sum(metrics['specificity_transfer_checks'][role]['frag3_only']['per_context'].values())
    table.append(f"| {names[role]} | {value['auc']:.4f} | {value['ap']:.4f} | {100*old['sensitivity']:.2f}% → {100*new['sensitivity']:.2f}% | {100*new['specificity']:.2f}% | {passed}/4 |")
summary_table = '\n'.join(table)
all_bytes = 13040412 + metadata['retained_body_bytes'] + pixels['retained_body_bytes']
assert all_bytes == 861973578
evidence_paths = {
    'benchmark': HERE.parent / 'fifteenth-experiment/BENCHMARK_CONTRACT.json',
    'research_design': HERE / 'RESEARCH_DESIGN.md',
    'load_accounting': HERE / 'LOAD_ACCOUNTING_CLARIFICATION.md',
    'production_design': HERE / 'PRODUCTION_INPUT_DESIGN.md',
    'metadata_result': HERE / 'production-acquisition/metadata-run/result.json',
    'pixel_result': HERE / 'production-acquisition/pixel-run/result.json',
    'raw_provenance': HERE / 'production-acquisition/pixel-run/raw-provenance.json',
    'pooled_descriptor': HERE / 'production-pooling-v2/pooling-run/pooled-descriptor.json',
    'pooling_result': HERE / 'production-pooling-v2/pooling-run/result.json',
    'independent_CT': HERE / 'numeric-review/production-verification/completed-review/verification.json',
    'inference_plan': HERE / 'inference/final-plan/comparison-plan.json',
    'inference_run': HERE / 'inference/run/results.json',
    'cutoff_barrier': HERE / 'inference/run/locks-ready.json',
    'metrics': HERE / 'inference/analysis/metrics.json',
    'original_header_STOP': HERE / 'retrieval/header-run/result.json',
    'original_pooling_STOP': HERE / 'pooling-original-attempt-stop.json',
    'pooling_correction': HERE / 'POOLING_JSON_CORRECTION.md',
    'original_inference_review_STOP': HERE / 'numeric-review/inference-verification/completed-review/FAILURE_RECEIPT.json',
    'retrospective_execution_context': HERE / 'inference-execution-context.json',
}
evidence = {key: record(path) for key, path in evidence_paths.items()}
evidence.update(independent_inference=independent_pin, presentation=presentation_pin)
contract = dict(schema_version=1, status='completed_independently_verified',
    created_at_utc=datetime.now(timezone.utc).isoformat(), evidence=evidence,
    model_execution=dict(tile_forwards=450, calibration_forwards=150, evaluation_forwards=300,
        **run['model_load_operations'], backward_calls=0, optimizer_updates=0, new_checkpoints=0,
        model_execution_attempts=1, model_execution_retries=0, wall_seconds=run['wall_seconds']),
    input_execution=dict(objects=153, contexts=6, shape=[17,384,384],
        production_metadata_GETs=450, production_pixel_GETs=1530, earlier_probe_GETs=21,
        all_phase_GETs=2001, all_phase_retained_response_body_bytes=all_bytes,
        production_pixel_response_body_bytes=pixels['retained_body_bytes'],
        useful_source_sample_bytes=812187648, automatic_network_retries=0,
        corrected_pooling_wall_seconds=pooled['wall_seconds'],
        whole_M_P_C_elapsed_seconds=pooled['production_elapsed_seconds'],
        whole_M_P_C_wall_cap_seconds=1500, combined_production_bytes=pooled['combined_production_bytes'],
        combined_production_cap_bytes=2*1024**3, pooling_attempts=2,
        original_pooling_STOP_before_numeric_CT_decode=True, bitwise_independent_CT_PASS=True),
    thresholds=metrics['thresholds'], paris2_equal_fragment=metrics['paris2_equal_fragment'],
    specificity_transfer_checks=metrics['specificity_transfer_checks'],
    promoted_model=None, adopted_baseline='unchanged strict seed42',
    interpretation='All three new Paris1 cutoff rules fail the all-four Paris2 specificity check; sensitivity remains low. Ranking varies strongly by region; candidates have slightly higher mean AUC but lower mean AP and sensitivity than baseline.',
    limitations=['One evaluation parent papyrus, Paris2, with two fragments and four contexts.',
        'Labels were inspected for support gates; evaluation is not label-blind.',
        'Complete upstream teacher/pretraining exposure is unknown.',
        'Exact CT bytes do not establish optical annotation or anatomical registration correctness.',
        'No independent-pixel uncertainty estimates, significance test, model promotion or authentic reading.',
        'Future method decisions informed by these results need fresh evaluation material for confirmation.',
        'Saved-output arithmetic and source/receipt review do not independently authenticate weight execution.'],
    paid_GPU_spend_GBP=0, total_first_GPU_pilot_cap_GBP=10,
    new_artifacts_uploaded_to_R2=False, publication=False, transcription=None, translation=None)
write_new(HERE / 'COMPLETION_CONTRACT.json', contract)

report = f'''# Plinius: Paris1-to-Paris2 calibration transfer

The fixed CPU comparison is complete and independently verified. **None of the three models passed the preregistered requirement of at least 95% specificity in all four Paris2 regions using its Paris1-only cutoff.** The unchanged baseline remains the comparator; no candidate is promoted.

{summary_table}

These are equal-context means within each fragment, then an equal-fragment mean across the two Paris2 fragments. Mean supplied-positive prevalence is 21.02%. A mean specificity above 95% does not satisfy the per-region requirement. All three fail in Frag1 context1; the other three regions pass. The old fourteenth cutoffs pass specificity in all four regions, with still lower sensitivity. New cutoffs were fitted only from the two fixed Frag3/Paris1 negative supports and locked for all three models before any Paris2 forward.

Baseline cutoffs were 0.7558947206 previously and 0.7266893387 from Paris1; A changed from 0.7450202703 to 0.7076279521; B from 0.7422624826 to 0.7000159621. Exact values, confusions, all metrics and per-region results are in [the tables](reporting/completed-results/tables.md) and [metrics JSON](inference/analysis/metrics.json).

## What this tells us

Baseline ROC AUC is 0.4461, 0.9037, 0.5013 and 0.7795 across the four Paris2 contexts. The large within-fragment variation matters more than the small candidate mean-AUC gains (+0.0040 and +0.0053). Both candidates have lower mean average precision and sensitivity than baseline. A cutoff change leaves score ranking unchanged.

In the failing Frag1 context1, baseline specificity is 93.07% and sensitivity 0.31% at its Paris1 cutoff: 61 true positives, 4,796 false positives, 64,363 true negatives and 19,339 false negatives. A and B identify none of the 19,400 supplied-positive pixels there, while producing 4,886 and 4,924 false positives respectively. The outcomes warrant a focused source/alignment/depth audit before more continuation training. They do not yet identify whether the cause is representation, surface/annotation registration, acquisition-domain shift or a combination.

## Execution and independent checks

The one model run took {run['wall_seconds']:.3f} seconds: 150 calibration and 300 evaluation tile forwards, six final-role worker constructions and ten strict state loads (six initializers plus four candidate replacements). There were zero backward calls, updates, new checkpoints or model retries. All three cutoff locks preceded evaluation. Independent saved-output reconstruction checked normalized-input hashes, all 18 score maps/450 captures, blending, coverage, cutoffs, confusions, ranking and macro metrics, and receipt ordering. It used no Torch, checkpoints or model execution.

All six float32[17,384,384] inputs were independently reconstructed bit-for-bit from the retained CT ranges; all 24 label/support arrays remain literal fifteenth artifacts. The source TIFFs use mixed byte orders, and Frag1 strip offsets vary between 260 and 262 bytes. Each of 153 files was checked separately.

The production pixel download used 1,530 GETs and {pixels['retained_body_bytes']:,} encoded response-body bytes in {pixels['wall_seconds']:.3f} seconds. Including the original/revised probes and the 450 new header requests, this phase retained {all_bytes:,} response-body bytes over 2,001 GETs. These are body counts, not total wire traffic. The narrow-row production body total is about 83% below the earlier 4,955,143,680-byte full-width-band payload projection; full scans were not downloaded. No automatic network retries occurred.

Corrected pooling took {pooled['wall_seconds']:.3f} seconds with {pooled['peak_rss_bytes']:,} bytes peak RSS. The whole M/P/C window, including planning and the correction, was {pooled['production_elapsed_seconds']:.3f} seconds under its 1,500-second cap. Producer and pooling outputs total {pooled['combined_production_bytes']:,} bytes under 2 GiB.

## Preserved technical stops

The first 8-byte header probe stopped on its little-endian-only assumption; a separately reviewed reader added explicit II/MM handling before further acquisition. The original pooling attempt stopped before numeric CT decoding because JSON tag keys were strings while reconstructed parser keys were integers. A separate V2 normalizes only those keys; all tag values and pooling arithmetic stayed exact. The first saved-output checker stopped before scientific array reads because it required absolute worker output arguments; the controller had retained valid relative arguments. Its separate correction resolves only that field against a hash-pinned, explicitly retrospective record of the actual root execution directory. Original sources, plans, failures and every scientific output remain retained.

## Limits and next step

Paris2 supplies one evaluation parent papyrus, not two independent scroll replications. Fixed support gates used labels; complete upstream model exposure and historical registration remain unresolved. Byte correctness does not authenticate the annotation or written surface. There are no pixel-independent confidence intervals, significance claims, first-in-class claim, transcription or translation.

The next recommended CPU task is [the source/alignment/depth diagnostic](NEXT_STUDY_DRAFT.md), using the fixed six contexts and retaining this result. Any subsequent method choice is informed by these Paris2 results and needs fresh evaluation material for a confirmatory claim. GPU spending remains **GBP 0**, with the **GBP 10 total first-pilot cap** unchanged. No new paid resource, upload, public redistribution, prize submission, new terms acceptance or outreach occurred. Retained data-use restrictions and EduceLab attribution continue to apply.

Exact completion evidence is bound in [COMPLETION_CONTRACT.json](COMPLETION_CONTRACT.json). Local retention and the environment startup draft are separate from backup, publication or a fresh-task restoration test.
'''
assert not (HERE / 'RESULTS.md').exists()
(HERE / 'RESULTS.md').write_text(report)
next_text = '''# Draft next study: explain regional failure before GPU training

This is a recommended follow-up, not an executed or GPU-ready experiment. The sixteenth result is retained unchanged: all three Paris1-calibrated cutoffs fail one Paris2 region, sensitivity is low, and baseline ranking varies markedly within fragments.

Use the existing six regions and saved raw bytes, labels, scores and source metadata for a bounded CPU source/alignment/depth audit. Check the declared optical-label transform, raster orientation and exposed-surface/depth provenance against independent source evidence. Summarize fixed depth/intensity and prediction-support diagnostics for every region; do not discard difficult regions. Where provenance is unavailable, retain that uncertainty explicitly.

Do not choose a new flip, depth window, crop, score inversion or operating cutoff because it improves the Paris2 results. Any physically justified adapter correction requires its own documented evidence, frozen contract and new result alongside the original. Any exploratory regional analysis must be labelled evaluation-informed. Keep Paris2 fragments and aliases excluded from gradient/teacher/cutoff fitting; secure fresh parent-level evaluation material before claiming a future method generalizes.

After this audit, select a concrete research hypothesis about depth/geometry-aware representation or nuisance robustness. Compare it with the unchanged seed42 baseline under an explicit training budget and success criteria. A useful GPU pilot needs a frozen experiment, pinned CUDA image/environment, current price, independent shutdown and verified bucket-scoped worker credentials. Preserve the GBP 10 TOTAL cap including failures, retries and attributable overheads. No paid resource is authorized by this draft. Further source requests need their own explicit bounded acquisition plan; no new terms, publication or upload is included.
'''
(HERE / 'NEXT_STUDY_DRAFT.md').write_text(next_text)
previous_path = LAB / 'reports/research-register-v11.json'
previous = json.loads(previous_path.read_text())
assert record(previous_path)['sha256'] == '0f7db66324b62699c1ed2c3b94d1742752fd3522e4012892fcc189e6a9b1ca2a'
assert len(previous['experiments']) == 15
register = dict(previous)
register.update(created_at_utc=datetime.now(timezone.utc).isoformat(), previous_register=record(previous_path),
    scope='Additive register after sixteenth fixed-weight calibration-transfer comparison; all prior studies unchanged',
    experiments=previous['experiments'] + [dict(id='sixteenth-experiment',
        purpose='Efficient exact CT acquisition and fixed-model Paris1-to-Paris2 calibration transfer',
        status='completed; all three primary specificity-transfer checks failed; independent CT and saved-output checks passed',
        report=record(HERE / 'RESULTS.md'), completion_contract=record(HERE / 'COMPLETION_CONTRACT.json'),
        metrics=metrics_pin, independent_verification=independent_pin,
        primary_specificity_transfer_all_four=metrics['primary_specificity_transfer_all_four'],
        paris2_equal_fragment=metrics['paris2_equal_fragment'], model_execution=contract['model_execution'],
        network=contract['input_execution'], promoted_model=None, upstream_independent=False,
        evaluation_parent_papyri=1, GPU_spend_GBP=0, GPU_pilot_total_cap_GBP=10,
        new_artifacts_uploaded_to_r2=False, transcription=None, translation=None)],
    next_study=dict(record(HERE / 'NEXT_STUDY_DRAFT.md'), status='Draft CPU source/alignment/depth audit; not executed'),
    readiness_check=dict(CPU_research='Exact six-input acquisition and fixed comparison independently verified; next source/alignment/depth diagnostic remains a draft',
        GPU_launch='Not ready: useful paid experiment not frozen; CUDA/pricing/shutdown and scoped worker delivery remain prerequisites',
        GPU_pilot_total_cap_GBP=10, GPU_spend_GBP=0, publication='none',
        configuration_save='Configuration draft tracked separately; no backup/publication/fresh-task restoration claim',
        tracked_repository_commit='49fabf3e890d5b880f391b27d581f1e7a1ebdbcc'))
register_path = LAB / 'reports/research-register-v12.json'
write_new(register_path, register)
assert json.loads(register_path.read_text())['experiments'][:15] == previous['experiments']
startup = f'''# Startup after the sixteenth fixed-model comparison

Use /workspace/Herculaneum at commit 49fabf3e890d5b880f391b27d581f1e7a1ebdbcc. Preserve its existing untracked .codex/ and keep tracked source unchanged. No worktree is needed. Helpers/data/results are in /workspace/herculaneum-lab; activate with `source /workspace/herculaneum-lab/activate.sh`. Python: /workspace/.venvs/herculaneum/bin/python. No service process is required.

Sixteenth completed. Read RESULTS.md and COMPLETION_CONTRACT.json here, contract SHA256 {record(HERE / 'COMPLETION_CONTRACT.json')['sha256']}. Register v12 SHA256 {record(register_path)['sha256']}. All six CT inputs were independently bitwise reconstructed. One unchanged three-model run performed 450 forwards; all three Paris1-only rules failed the all-four Paris2 specificity check, with low sensitivity. Keep seed42 unchanged; no promotion, authenticated text or translation. Both Paris2 fragments are one evaluation parent; label-gate and unknown upstream exposure limitations remain.

Before further work, run this directory's verify_retained_state.py with --expected-manifest-sha256 and the literal external anchor in the environment startup instructions. It verifies every retained record and exact checkout identity/status read-only. Do not overwrite prior outputs, regenerate a manifest to hide a mismatch, or repeat completed acquisition/inference to replace an outcome. Missing evidence is a restoration blocker; configuration text is not an artifact backup.

Original little-endian header STOP, pre-decode JSON-key pooling STOP, and command-path saved-output-review STOP are preserved. Separately reviewed corrections retained original bytes, arithmetic and scientific outputs. The final independent saved-output check uses a hash-pinned retrospective root execution-context record solely to resolve the recorded relative output paths. Do not describe it as an independent launch-time trace.

Next recommendation is NEXT_STUDY_DRAFT.md: bounded CPU source/alignment/depth diagnosis of all six fixed regions before new training. Paris2 results now inform future reasoning; any subsequent method claim needs fresh evaluation material. No orientation/depth/crop/cutoff search is authorized by the old protocol. Keep evaluation fragments and aliases out of gradient, teacher and cutoff fitting.

GPU spending GBP 0; total first-pilot cap GBP 10 including retries/failures/overheads persists. No paid launch before a useful frozen experiment, pinned CUDA environment, current pricing, independent shutdown and verified bucket-scoped worker delivery. Last local handoff did not prove Runpod secret import or worker injection; temporary R2 credentials expire conservatively 2026-10-15 09:58:59 UTC. No credentials are needed for the completed CPU work; never request raw keys in chat. Retained legacy data-use restrictions and EduceLab citations apply. No upload/publication/submission/new terms/outreach occurred.

Saving startup instructions persists a draft; it does not itself publish, back up research artifacts, restart processes, import credentials or demonstrate fresh-task restoration. Preserve all unrelated configuration settings.
'''
(HERE / 'STARTUP_AFTER_SIXTEENTH.md').write_text(startup)
print(json.dumps(dict(status='PASS', contract=record(HERE / 'COMPLETION_CONTRACT.json'),
    report=record(HERE / 'RESULTS.md'), register=record(register_path),
    startup=record(HERE / 'STARTUP_AFTER_SIXTEENTH.md'), prior_experiments_unchanged=15)))
