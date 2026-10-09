"""Record phase22 findings and preserve preceding register entries byte-for-value."""
from pathlib import Path
import datetime
import hashlib
import json

P=Path(__file__).resolve().parent
R=P.parent

def pin(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)
def read(rel): return json.loads((P/rel).read_text())
def write(path,data):
    with path.open('x') as f: json.dump(data,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')

seals={
 'benchmark-inventory/retention-manifest.json':'03c6fa6a8710004a6e16482cc72e5a5b51f0b19fc01f30545ac2676d276bf211',
 'benchmark-metadata/retention-manifest.json':'19b71304ca855785fe8c11929c067aa50cc7b40b6fb43bea9d49689ead30f1a8',
 'benchmark-metadata/v2/retention-manifest.json':'c66f7173e137911ecb35abbc655278cfb364078f2083ec96df40bfd77e131c2a',
 'native-input-contract/RETENTION_MANIFEST.json':'9bd47e62228cbb46dd12fbff2348408121a36a2e4bd816ecd607fd9faf1cea7a',
 'checkpoint-qualification/RETENTION_MANIFEST.json':'af9f06ea04a87dfc27a7583854af5196a8215836ea44ce895fe607c378a131eb',
 'checkpoint-review/RETENTION_MANIFEST.json':'4d862b49f9c4e456d8044ef26563ef55e1672ba42e767ec0c64fdca79dad71b9',
 'model-parity/RETENTION_MANIFEST.json':'ec50b335503feaea3897308120b32b58cd4f0a3e6461179f66a2947b09aded6b',
 'parity-review/RETENTION_MANIFEST.json':'7171e64dc17ff4c1969cecf8d022e88eefa9fa3c2fb1e4bdd024c2f41fb3bce9',
}
verified=[]
for rel,expected in seals.items():
    f=P/rel; assert pin(f)['sha256']==expected
    d=json.loads(f.read_text()); records=d.get('files',d.get('records')); assert isinstance(records,list)
    for item in records:
        path=Path(item['path']); path=path if path.is_absolute() else f.parent/path
        actual=pin(path)
        assert actual['sha256']==item['sha256']
        assert actual['size_bytes']==item.get('size_bytes',item.get('byte_length'))
    verified.append(dict(manifest=pin(f),records_verified=len(records)))
parity=read('parity-review/verification-v1.json'); assert parity['status']=='PASS'
assert all(x[k]['exact_bitwise'] and x[k]['max_absolute']==0 for x in parity['cross_factory'] for k in ('logits','probabilities','raw_score'))
comparison_counts={k:sum(x[k]['count'] for x in parity['cross_factory']) for k in ('logits','probabilities','raw_score')}
assert comparison_counts==dict(logits=180224,probabilities=180224,raw_score=167936)
controller=read('model-parity/run-v1/controller-result.json'); assert controller['status']=='PASS' and controller['model_forwards']==22
stages=[]
for name,rel,countkey,byteskey in [
 ('historical-source','native-input-contract/source/request-ledger.json','attempts','observed_body_bytes'),
 ('historical-activation','native-input-contract/activation-source/receipt.json','attempts','observed_body_bytes'),
 ('native-checkpoint','checkpoint-qualification/run-v1/ledger.json','attempts','observed_body_bytes'),
 ('metadata-direct-failure','benchmark-metadata/run-v1/request-ledger.json','requests','total_observed_body_bytes'),
 ('metadata-proxy-correction','benchmark-metadata/v2/run-v1/request-ledger.json','requests','total_observed_body_bytes')]:
    d=read(rel); count=d[countkey]; count=len(count) if isinstance(count,list) else count
    stages.append(dict(name=name,requests=count,observed_response_body_bytes=d[byteskey],receipt=pin(P/rel)))
requests=sum(x['requests'] for x in stages); netbytes=sum(x['observed_response_body_bytes'] for x in stages)
assert requests==26 and netbytes==273128944,(requests,netbytes)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
root_review=dict(status='PASS',created_at_utc=now,subdirectory_seals=verified,
    report=pin(P/'RESULTS.md'),next_study=pin(P/'NEXT_STUDY_DRAFT.md'),
    exact_cross_factory_counts=comparison_counts,
    scientific_interpretation='Strict loading and named synthetic CPU compatibility only. No accuracy, CUDA equivalence, teacher independence or promotion.',
    metadata_interpretation='Exact complete packages omit mapping validity and linked provenance; no claim of global absence. Preserved v1 failure plus explicit revised proxy transport.',
    next_design='Fixed native-reference comparison on reused development; no training/tuning or independent-confirmation claim')
write(P/'ROOT_REPORT_REVIEW.json',root_review)
evidence={name:pin(P/rel) for name,rel in {
 'report':'RESULTS.md','next_study':'NEXT_STUDY_DRAFT.md','root_review':'ROOT_REPORT_REVIEW.json',
 'inventory':'benchmark-inventory/ELIGIBILITY_MATRIX.json',
 'metadata':'benchmark-metadata/v2/PACKAGE_FINDINGS.json',
 'input_contract':'native-input-contract/INPUT_CONTRACT.json',
 'source_closure':'native-input-contract/HISTORICAL_FACTORY_CLOSURE.json',
 'checkpoint_author':'checkpoint-qualification/run-v1/qualification.json',
 'checkpoint_review':'checkpoint-review/verification-v1.json',
 'parity_plan':'model-parity/PLAN.json','parity_controller':'model-parity/run-v1/controller-result.json',
 'parity_review':'parity-review/verification-v1.json'}.items()}
contract=dict(status='completed_verified_native_checkpoint_and_synthetic_cpu_compatibility',created_at_utc=now,
 adopted_baseline='unchanged strict seed42',promoted_model=None,
 conclusion='Full native integrity, strict loading and frozen nine-case CPU compatibility PASS; no independent accuracy reserve established.',
 decision='One fixed native-reference comparison on existing development inputs next; no further tuning or GPU scaling yet.',
 model_constructions=2,strict_state_loads=2,synthetic_model_forwards=22,scientific_model_forwards=0,
 scientific_optimizer_updates=0,new_candidate_scores=0,cross_factory_bitwise_identical=True,
 cross_factory_counts=comparison_counts,stage2_controller_wall_seconds=controller['wall_seconds'],
 checkpoint_bytes=272767904,checkpoint_sha256='d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d',
 named_state_tensors=508,named_elements=68175426,actual_factory_groups=264,alias_pairs=244,unique_parameters=34546498,
 new_independent_accuracy_reserve=False,all_six_Paris_contexts='reused development',
 reserved_unscored_parents=['PHerc0009B','PHerc0500P2','PHerc0841','PHerc0051 / Frag6'],
 Frag6_stage_C_authorized=False,evidence=evidence,network_stages=stages,
 network_requests=requests,network_response_body_bytes=netbytes,automatic_retries=0,
 explicit_revised_acquisition='One metadata transport correction after preserved zero-body failure; separate root GO',
 GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,R2_worker_secret_delivery_verified=False,
 new_artifacts_uploaded_to_r2=False,publication=None,submission=None,transcription=None,translation=None)
write(P/'COMPLETION_CONTRACT.json',contract)
previous=R/'research-register-v17.json'
assert pin(previous)['sha256']=='3af78e067cf92884c3d9ac4d5bed4c491e2df6e1c22ec676448fd9d4d6a3ae8c'
register=json.loads(previous.read_text()); old_entries=json.loads(json.dumps(register['experiments']))
assert len(old_entries)==21
register['experiments'].append(dict(id='twentysecond-experiment',purpose='Native checkpoint/input qualification and benchmark provenance audit',
 completion_contract=pin(P/'COMPLETION_CONTRACT.json'),report=pin(P/'RESULTS.md'),
 numerical_verification=pin(P/'parity-review/verification-v1.json'),
 **{k:contract[k] for k in ('status','conclusion','decision','promoted_model','GPU_spend_GBP','GPU_pilot_total_cap_GBP','network_requests','network_response_body_bytes','new_candidate_scores','scientific_optimizer_updates','synthetic_model_forwards','new_artifacts_uploaded_to_r2','transcription','translation')},
 independent_confirmation_ready=False,synthetic_compatibility_passed=True))
assert register['experiments'][:-1]==old_entries
register.update(created_at_utc=now,scope='Additive register after phase22; first21 studies unchanged',previous_register=pin(previous),
 next_study={**pin(P/'NEXT_STUDY_DRAFT.md'), 'status':'Design recommendation; freeze separate scientific execution protocol before scoring'})
register['readiness_check'].update(CPU_research='Native checkpoint strict loading and nine-case synthetic CPU compatibility independently verified; ready for a separately frozen reference comparison',GPU_launch='Not ready: useful workload, pinned CUDA, current pricing, independent shutdown and actual scoped worker delivery required')
write(R/'research-register-v18.json',register)
print(json.dumps(dict(status='PASS',completion=pin(P/'COMPLETION_CONTRACT.json'),register=pin(R/'research-register-v18.json'),network_requests=requests,network_bytes=netbytes,subseals_verified=len(seals))))
