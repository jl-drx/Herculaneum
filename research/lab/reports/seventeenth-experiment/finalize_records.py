"""Bind the completed phase17 diagnostic and add a research-register entry."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent
def record(path):
    p=Path(path).resolve(strict=True)
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(path, expected=None):
    p=HERE/path
    r=record(p)
    if expected:assert r['sha256']==expected, str(p)
    return json.loads(p.read_text())
def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

review=read('independent-review/verification.json','b387a45228cb9c209510edfad70c05d1cfca17b61bc7432942a59c5f4d383347')
assert review['status']=='PASS'
ct=read('implementation/ct-run/ct-diagnostics.json','6e3930f01b2310f479c92d206cadd514e5b48fc7497ed7518b8634a670629fbf')
ir=read('implementation/ir-run/ir-diagnostics.json','4083f0b3100eb35a512655252b0d3c37a2c0239ce683daf26ee9ddd5aead2b27')
ppm=read('implementation/ppm-run/ppm-diagnostics.json','2d75cc90265a8f64eb11a30c2283783e54225e399147e18039203a192df853a6')
overview=read('overview-v2/summary.json')
assert all(x['status']=='PASS' and x['model_calls']==x['network_calls']==x['checkpoint_reads']==0 for x in [ct,ir,ppm])
images=[record(HERE/c['panel']['path']) for data in [ct,ir] for c in data['cases']]
images.append(record(HERE/'overview-v2/regional-overview.png'))
save(HERE/'ROOT_VISUAL_REVIEW.json',dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    status='completed',reviewer='root',images=images,
    scope='Visual inspection of all six fixed CT and all six IR panels plus final overview; no fitted alignment or new numerical threshold',
    observations=['No gross visible IR-label raster swap in the six fixed panels.',
                  'Native and pooled fixed slices preserve the same spatial structures.',
                  'All six fixed cross-sections show structured material around/below nominal center and reduced variation at larger indices.',
                  'These views do not independently quantify CT-to-IR warp residual or locate microscopic ink.'],
    earlier_overview_retained='overview/; overview-v2 changes legend layout only'))
assert read('overview/summary.json')==overview, 'Presentation-only revision changed numbers'
evidence={k:record(HERE/p) for k,p in {
    'design':'RESEARCH_DESIGN.md','preexecution_clarifications':'PREEXECUTION_CLARIFICATIONS.md',
    'predecode_amendment':'PPM_PREDECODE_AMENDMENT.md','zero_mapping_clarification':'PPM_ZERO_MAPPING_CLARIFICATION.md',
    'source_manifest':'source-review/MANIFEST.json','source_fetch':'source-review/PPM_SOURCE_RECEIPT.json',
    'header_acquisition':'acquisition/header-run/result.json','sample_acquisition':'acquisition/sample-run/result.json',
    'ct':'implementation/ct-run/ct-diagnostics.json','ir':'implementation/ir-run/ir-diagnostics.json',
    'ppm':'implementation/ppm-run/ppm-diagnostics.json','producer_seal':'implementation/FINAL_RECEIPT.json',
    'producer_runtime':'implementation/runtime-supplement.json','independent_review':'independent-review/verification.json',
    'independent_seal':'independent-review/retention-manifest.json','root_visual_review':'ROOT_VISUAL_REVIEW.json',
    'summary':'overview-v2/summary.json','presentation':'overview-v2/presentation-receipt.json',
    'report':'RESULTS.md','next_study':'NEXT_STUDY_DRAFT.md'}.items()}
network=dict(public_GETs=22,retained_response_body_bytes=41827766,historical_reader_GETs=1,historical_reader_body_bytes=7494,
             IR_and_prefix_GETs=9,IR_and_prefix_body_bytes=41044172,sparse_GETs=12,sparse_body_bytes=776100,
             useful_PPM_bytes=419904,full_PPM_downloads=0,new_CT_downloads=0,automatic_retries=0,
             scope='Response-body accounting; does not include wire/TLS/header overhead')
contract=dict(schema_version=1,status='completed_independently_verified',created_at_utc=datetime.now(timezone.utc).isoformat(),
    purpose='Evaluation-informed regional alignment and depth diagnostic of all six fixed contexts',
    conclusion='No clear implementation/alignment/depth fault explains the weak CT regions; physical registration residual and exact ink depth remain unresolved.',
    evidence=evidence,network=network,
    numerical_execution={k:{field:d[field] for field in ['command','wall_seconds','cpu_seconds','peak_rss_bytes','runtime','model_calls','checkpoint_reads','network_calls','gpu_seconds']} for k,d in [('CT',ct),('IR',ir),('PPM',ppm)]},
    independent_checks={k:review[k] for k in ['ct','ir','ppm','coverage']},
    per_case_summary=overview['cases'],
    fixed_PPM_sampling=dict(result_centers=486,valid_result_centers=478,unmapped_result_centers=8,
        primary_result_centers=409,unmapped_primary_result_centers=0,flat_centers=486,
        interpretation='All six maps uniquely little-endian under fixed geometry invariants; not authenticated historical hardware'),
    failed_checker_attempts_preserved=2,numerical_producer_failures=0,
    adopted_baseline='unchanged strict seed42',promoted_model=None,new_forwards=0,backward_calls=0,new_checkpoints=0,
    paid_GPU_spend_GBP=0,total_first_GPU_pilot_cap_GBP=10,new_artifacts_uploaded_to_R2=False,publication=False,
    transcription=None,translation=None,
    limitations=['Registered IR and traced labels share provenance; their agreement does not independently validate CT-to-IR alignment.',
                 'Sparse PPM geometry cannot certify the whole surface, native-volume frame or anatomical normal sign.',
                 'Current renderer midpoint is not a hash bridge to the historical sampling invocation.',
                 'Native depths7..57 only; profiles cannot locate authenticated ink or tissue outside the acquired extent.',
                 'Paris2 is one parent papyrus; labels were already inspected and upstream training exposure is incomplete.',
                 'Evaluation-informed method changes require fresh confirmatory material; no shifts/depths/crops/cutoffs fitted.',
                 'Retention and draft configuration are not backup or fresh-task restoration.'])
save(HERE/'COMPLETION_CONTRACT.json',contract)
prior=LAB/'reports/research-register-v12.json'
assert record(prior)['sha256']=='e3b311c4cf881e7d85453b4fe22fa36482844870a4e85fbef1b9ec736ed378b4'
register=json.loads(prior.read_text());assert len(register['experiments'])==16
register['previous_register']=record(prior)
register['created_at_utc']=datetime.now(timezone.utc).isoformat()
register['scope']='Additive register after seventeenth regional alignment/depth diagnostic; prior16 studies unchanged'
register['experiments'].append(dict(id='seventeenth-experiment',purpose=contract['purpose'],status=contract['status'],
    conclusion=contract['conclusion'],completion_contract=record(HERE/'COMPLETION_CONTRACT.json'),report=evidence['report'],
    independent_verification=evidence['independent_review'],summary=evidence['summary'],network=network,
    model_calls=0,checkpoint_reads=0,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,promoted_model=None,
    new_artifacts_uploaded_to_r2=False,evaluation_informed=True,transcription=None,translation=None))
register['next_study']={**evidence['next_study'],'status':'Draft controlled depth-aware representation experiment; not executed or GPU-ready'}
register['readiness_check']['CPU_research']='Regional CT/IR/PPM diagnostic independently verified; no clear alignment/depth fault established'
register['readiness_check']['GPU_launch']='Not ready: matched representation experiment/fresh evaluation not frozen; CUDA/pricing/shutdown/scoped worker delivery remain prerequisites'
save(LAB/'reports/research-register-v13.json',register)
startup=f'''# Startup after the seventeenth regional diagnostic

Use /workspace/Herculaneum at commit49fabf3e890d5b880f391b27d581f1e7a1ebdbcc; preserve untracked .codex/ and tracked source. Lab: /workspace/herculaneum-lab; activate.sh selects /workspace/.venvs/herculaneum/bin/python. No service, worktree or new installation is needed.

Phase17 is complete. Read RESULTS.md and COMPLETION_CONTRACT.json in reports/seventeenth-experiment (contract SHA256 {record(HERE/'COMPLETION_CONTRACT.json')['sha256']}). Registerv13 SHA256 {record(LAB/'reports/research-register-v13.json')['sha256']}. Independently verified exact CT, six IR regions and sparse PPM geometry found no clear fault explaining weak regions. IR agreement is label-derived; physical registration residual and exact ink depth remain unmeasured. No model rerun, promotion or authenticated text. Seed42 unchanged.

Before new work run verify_retained_state.py here with --expected-manifest-sha256 and the literal anchor saved in environment startup instructions. Do not overwrite prior evidence or regenerate a manifest to conceal a mismatch. Missing artifacts require restoration; startup text is not a backup. All prior16 evidence, two checker schema stops, unexecuted header-planv1 and presentationv1 are retained.

Actual numerical producers used platform Python3.12.14/NumPy2.3.5/SciPy1.17.0; independent review used lab venv NumPy2.5.3. Both are documented and comparisons passed. Reproduce with the recorded executable, not an assumed shell python. For plots set MPLCONFIGDIR and XDG_CACHE_HOME to writable /tmp locations.

NEXT_STUDY_DRAFT.md proposes one controlled depth-aware representation change with raw-intensity controls. Do not optimize Paris2 shifts, depths, crops or cutoffs. Keep its fragments/aliases out of gradient/teacher/cutoff fitting; secure fresh material for confirmation. Sparse endian resolution is conditional on declared invariants, not historical hardware or anatomy.

GPU spend GBP0; first-pilot TOTAL cap GBP10 persists. No paid launch before a frozen useful experiment, pinned CUDA environment, current pricing, independent shutdown and verified bucket-scoped worker delivery. Secret import/injection remains unverified; temporary credentials expire2026-10-15 09:58:59UTC. Never request keys in chat. No upload/publication/new terms/outreach. EduceLab attribution and retained use restrictions apply.

Saving startup instructions persists a draft, not publication, backup, process restart or proven fresh-task restoration. Preserve unrelated configuration settings.
'''
with (HERE/'STARTUP_AFTER_SEVENTEENTH.md').open('x') as f:f.write(startup)
print(json.dumps(dict(status='PASS',contract=record(HERE/'COMPLETION_CONTRACT.json'),
    register=record(LAB/'reports/research-register-v13.json'),startup=record(HERE/'STARTUP_AFTER_SEVENTEENTH.md'))))
