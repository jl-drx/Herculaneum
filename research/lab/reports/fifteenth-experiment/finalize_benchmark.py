"""Assemble descriptive benchmark records after successful independent verification."""
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent.parent

def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, value):
    path = Path(path)
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n' if not isinstance(value,str) else value)

p=argparse.ArgumentParser()
p.add_argument('--independent-review', required=True)
a=p.parse_args()
independent=Path(a.independent_review).resolve(strict=True)
review=read(independent)
assert review['status']=='PASS'
source=read(ROOT/'SOURCE_CONTRACT.json')
plan=ROOT/'implementation/final-plan/label-plan.json'
acquisition=ROOT/'implementation/acquisition/provenance.json'
panel=ROOT/'implementation/selection/panel-provenance.json'
acq,selection=read(acquisition),read(panel)
for name, path in [('plan', plan), ('acquisition', acquisition), ('panel', panel)]:
    assert review['inputs'][name] == record(path), 'Independent review covers different inputs'
assert review['selected_array_count'] == 24 and review['all_selected_arrays_exact'] is True
assert review['all_candidates_counts_and_integer_gates_exact'] is True
assert review['roles_parents_alias_exclusions_verified'] is True
assert acq['status']==selection['status']=='PASS'
assert len(acq['objects'])==6 and len(selection['selected'])==6
workers=[read(w['result']['path']) for w in selection['workers']]
assert all(w['status']=='PASS' and len(w['selected'])==2 for w in workers)
assert [(w['fragment'],w['identity']['role'],w['identity']['parent_papyrus']) for w in workers]==[
    ('frag1','evaluation','Paris2'),('frag2','evaluation','Paris2'),('frag3','calibration','Paris1')]
rows=[]
for w in workers:
    for c in w['selected']:
        for f in c['files'].values():
            r=record(f['path']); assert r['sha256']==f['sha256'] and r['size_bytes']==f['bytes']
        rows.append('| '+ ' | '.join([c['case_id'],c['role'],c['parent_papyrus'],str(c['bbox_source_xywh']),f"{c['primary_positive']:,}",f"{c['primary_negative']:,}"])+ ' |')
contract=dict(schema_version=1,status='PASS: label-supported benchmark constructed and independently verified',
    created_at_utc=datetime.now(timezone.utc).isoformat(),research_series_working_name='Plinius',
    source_contract=record(ROOT/'SOURCE_CONTRACT.json'),design=record(ROOT/'BENCHMARK_DESIGN.md'),
    plan=record(plan),acquisition=record(acquisition),panel=record(panel),independent_verification=record(independent),
    selected_contexts=selection['selected'],fragments=[dict(id=w['identity']['id'],aliases=[w['identity']['legacy_alias']],
        parent_papyrus=w['identity']['parent_papyrus'],role=w['identity']['role'],
        role_scope='Whole physical fragment, aliases and all scan versions',
        prohibited_uses=(['gradient training','teacher targets','cutoff fitting','normalization parameter fitting',
                         'crop/depth/orientation search','model selection'] if w['identity']['role']=='evaluation' else []),
        quality=w['quality'],selection_result=w2['result']) for w,w2 in zip(workers,selection['workers'])],
    statistical_units=dict(physical_fragments=3,parent_papyri=2,evaluation_fragments=2,evaluation_parent_papyri=1,
        calibration_parent='Paris1',evaluation_parent='Paris2',existing_seen_diagnostic='Frag4 / PHercParis1Fr39 / Paris1'),
    exposure=dict(local_previous_Frag1_to_3_scientific_use='None found in retained first-fourteenth records',
        evaluation_labels_blinded=False,upstream_independent=False,complete_teacher_pretraining_history='unknown'),
    sampling='Deterministic label-supported regions, not random whole-fragment prevalence/performance sampling',
    limitations=['Optical registration residual unmeasured','Historical publication-to-current-PNG hash bridge unavailable',
                 'Conservative33-square support is not an exact paper FIND_EDGES reproduction',
                 'Source-canvas contexts are disjoint; native3D disjointness unmeasured'],
    deferred=source['deferred_candidates'],license=source['license'],
    execution=dict(metadata_requests=source['metadata_requests'],label_png_requests=acq['http_attempts'],
        encoded_label_png_bytes=acq['response_body_bytes'],CT_bytes=0,model_calls=0,backward_calls=0,
        new_checkpoints=0,scientific_retries=0,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10),
    CT_ready=False,accuracy_measured=False,promoted_model=None,transcription=None,translation=None,
    next_study=record(ROOT/'NEXT_STUDY_DRAFT.md'))
write_new(ROOT/'BENCHMARK_CONTRACT.json',contract)
report=f'''# Plinius: expanded optical-reference benchmark

The frozen label-only construction passed. Three additional physical fragments now supply six deterministically selected regions with separate calibration and evaluation roles. An independent implementation reconstructed the support and selections from the six source PNGs. This is benchmark preparation; no CT pixels or model predictions were acquired in this stage, and no accuracy or text-reading result is claimed.

| Region | Role | Parent | Native XYWH | Primary positives | Primary negatives |
|---|---|---|---|---:|---:|
'''+ '\n'.join(rows)+f'''

Each region is1152×1152 native pixels, sampled into384×384 labels. Primary support excludes a32-coarse-pixel context border, uncertain33×33 native neighborhoods and pixels outside the surface mask. Every selected region passed the frozen90% surface,512-positive and8192-negative gates. Selection used the first eligible grid region and the last eligible region separated by at least2304 native y pixels; no region, grid or gate was substituted after inspection. All24 arrays have retained byte hashes.

Frag3 (PHercParis1Fr34) is calibration. Frag1 (PHercParis2Fr47) and Frag2 (PHercParis2Fr143) are evaluation, reserved as whole fragments across aliases and scan versions. Both evaluation fragments share Paris2, giving one evaluation parent, distinct from calibration parent Paris1. Frag4 remains a previously seen Paris1 diagnostic. Evaluation labels were inspected for predefined support gates and are not blinded. Unknown upstream teacher/pretraining exposure prevents an upstream-independent generalization claim.

The source audit made42 bounded metadata requests totaling116,634 response-body bytes. Six complete PNG requests then downloaded{acq['response_body_bytes']:,} encoded bytes, with no retries or redirects. The selection controller took{selection['wall_seconds']:.3f} seconds in total, with maximum measured worker RSS{max(w['peak_rss_bytes'] for w in workers)/2**20:.1f} MiB. The exact candidate tables, source encodings, unsupported-positive counts and selected array identities are retained in the machine-readable contract and worker receipts. Independent verification is linked below; synthetic preparation failures were retained and corrected before real downloads.

The next useful experiment is an efficient CT reader check, followed by a fixed-weight calibration-transfer comparison. The old full-width TIFF-band reader would fetch about4.6 GiB for these six regions under the old layout assumption. The new TIFF layouts, Range behavior and stable object identities have not been probed. Freeze a bounded header/IFD audit and a transfer strategy before downloading CT. Then compare the unchanged baseline and both thirteenth states using only Frag3 to fit any new cutoffs, locking them before Paris2 evaluation. Do not start more training or rent a GPU based on benchmark construction alone.

Current legacy data-use terms match the retained version; redistribution/publication restrictions and EduceLab citations remain attached. No upload, publication, submission or new terms acceptance occurred. GPU spend is GBP0; the total first-pilot cap remains GBP10. Worker secret delivery remains unverified. Repository tracked files and all prior frozen research evidence remain unchanged.

Records: [benchmark contract](BENCHMARK_CONTRACT.json), [frozen design](BENCHMARK_DESIGN.md), [independent verification]({independent}), [CT cost calculation](implementation/planning/CT_COST.md), [next study](NEXT_STUDY_DRAFT.md).
'''
for old,new in [('is1152','is 1152'),('into384','into 384'),('a32','a 32'),('uncertain33','uncertain 33'),('frozen90','frozen 90'),(',512',', 512'),('and8192','and 8192'),('least2304','least 2304'),('All24','All 24'),('made42','made 42'),('totaling116','totaling 116'),('downloaded'+str(acq['response_body_bytes']), 'downloaded '+str(acq['response_body_bytes'])),('downloaded{','downloaded {'),('took'+f"{selection['wall_seconds']:.3f}",'took '+f"{selection['wall_seconds']:.3f}"),('RSS'+f"{max(w['peak_rss_bytes'] for w in workers)/2**20:.1f}",'RSS '+f"{max(w['peak_rss_bytes'] for w in workers)/2**20:.1f}"),('about4.6','about 4.6'),('GBP0','GBP 0'),('GBP10','GBP 10')]:report=report.replace(old,new)
report=report.replace('downloaded'+f"{acq['response_body_bytes']:,}",'downloaded '+f"{acq['response_body_bytes']:,}")
write_new(ROOT/'RESULTS.md',report)
previous=LAB/'reports/research-register-v10.json'
reg=read(previous)
old_entries=list(reg['experiments'])
reg.update(created_at_utc=datetime.now(timezone.utc).isoformat(),previous_register=record(previous),
    scope='Additive register after fifteenth label-only benchmark construction; every prior study unchanged',
    next_study=dict(**record(ROOT/'NEXT_STUDY_DRAFT.md'),status='Draft future CT reader and fixed-weight comparison; not executed'))
reg['experiments'].append(dict(id='fifteenth-experiment',status='completed: expanded label-supported benchmark; no model execution',
    purpose='Separate calibration/evaluation by parent papyrus before later predictions',
    benchmark_contract=record(ROOT/'BENCHMARK_CONTRACT.json'),report=record(ROOT/'RESULTS.md'),
    independent_verification=record(independent),**contract['execution'],
    physical_fragments=3,parent_papyri=2,evaluation_parent_papyri=1,selected_contexts=6,selected_arrays=24,
    upstream_independent=False,CT_ready=False,promoted_model=None,transcription=None,translation=None))
assert reg['experiments'][:-1]==old_entries
reg['readiness_check']['CPU_research']='Current-instance label-only construction and independent reconstruction passed; next CT layout/transfer audit remains unfrozen and unexecuted'
reg['readiness_check']['GPU_launch']='Not ready: no frozen useful paid experiment; worker secret delivery unverified; efficient CT input acquisition precedes GPU work'
write_new(LAB/'reports/research-register-v11.json',reg)
startup=f'''# Startup after fifteenth benchmark construction

Use existing /workspace/Herculaneum checkout at49fabf3e890d5b880f391b27d581f1e7a1ebdbcc. Do not create a worktree unless the user requests it. Tracked repository files stay unchanged; preserve its pre-existing untracked .codex/. Helpers and research outputs live in /workspace/herculaneum-lab. Activate with `source /workspace/herculaneum-lab/activate.sh`; Python is /workspace/.venvs/herculaneum/bin/python. No service process is required.

The fifteenth label-only benchmark is complete. Read RESULTS.md and BENCHMARK_CONTRACT.json here. Exact contract SHA256:{record(ROOT/'BENCHMARK_CONTRACT.json')['sha256']}; register v11 SHA256:{record(LAB/'reports/research-register-v11.json')['sha256']}. Three new physical fragments, two parent papyri, six regions: Frag3/Paris1 calibration and Frag1+Frag2/Paris2 evaluation. Both evaluation fragments are one parent group. Frag4 is seen Paris1 diagnostic. Whole evaluation fragments and aliases remain excluded from training, teacher targets, cutoff fitting, normalization parameter fitting, search and model selection. Unknown upstream exposure and inspected selection labels preclude upstream-independent or label-blind claims.

Before new work, run this directory's verify_retained_state.py with --expected-manifest-sha256 and the external anchor saved in the environment start instructions. It checks all retained files read-only and exact checkout identity/status. Do not overwrite prior outputs, rerun completed acquisition/selection into new outcomes, or regenerate the final manifest to conceal a mismatch. A missing workspace artifact is a restoration blocker; this draft is not an external artifact backup.

Next study remains a draft in NEXT_STUDY_DRAFT.md. No new CT pixels or model calls occurred here. Resolve TIFF layout/Range identity and efficient bounded transfer before acquiring CT; the old band reader's roughly4.6 GiB projection is not a new-header observation. Then freeze the unchanged three-model comparison and Frag3-only cutoff locking before Paris2 scoring. There is no promoted model, novel-method claim, transcription or translation.

GPU spend GBP0; GBP10 TOTAL first-pilot cap including failures/retries/overheads persists. No paid launch before frozen experiment, pinned CUDA environment, current pricing, independent shutdown and verified bucket-scoped worker delivery. Last local handoff did not establish Runpod secret import. Temporary R2 credentials expire conservatively2026-10-15 09:58:59 UTC; do not log values or request them in chat. No credentials needed for bounded public-source CPU work. No upload/publication/submission/new terms acceptance. Current release restrictions and citations remain attached.

Saving startup instructions persists a configuration draft only. It does not publish, back up artifacts, restart processes, import secrets or establish fresh-task restoration. Preserve all other saved configuration fields.
'''
for old,new in [('at49','at 49'),('SHA256:','SHA256: '),('roughly4.6','roughly 4.6'),('GBP0','GBP 0'),('GBP10','GBP 10'),('conservatively2026','conservatively 2026')]:startup=startup.replace(old,new)
write_new(ROOT/'STARTUP_AFTER_FIFTEENTH.md',startup)
print(json.dumps(dict(contract=record(ROOT/'BENCHMARK_CONTRACT.json'),register=record(LAB/'reports/research-register-v11.json'),report=record(ROOT/'RESULTS.md'))))
