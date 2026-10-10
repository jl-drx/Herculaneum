"""Cross-check independent, producer and transport receipts; no tensor reads."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace/herculaneum-lab/reports/twentysecond-experiment')
REVIEW=ROOT/'checkpoint-review'
RUN=ROOT/'checkpoint-qualification/run-v1'
def pin(p):return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p):return json.loads(p.read_text())
freeze=load(REVIEW/'CHECKER_FREEZE.json')
for r in freeze['files']:assert pin(Path(r['path']))==r
ind=load(REVIEW/'verification-v1.json');prod=load(RUN/'qualification.json');ledger=load(RUN/'ledger.json');controller=load(RUN/'controller.json')
release=load(ROOT/'CHECKPOINT_INDEPENDENT_GO.json')
assert pin(RUN/'qualification.json')==release['qualification']
assert ind['status']=='PASS' and controller['status']=='PASS'
assert prod['status']=='FULL_FILE_INTEGRITY_AND_VALUES_QUALIFIED_LOAD_UNVERIFIED'
assert prod['checkpoint']==ind['artifact']==release['artifact']
assert prod['tensor_name_count']==ind['named_tensor_count']==508
assert prod['finite_named_elements']==ind['named_numel']==68175426
assert prod['alias_pair_count']==ind['alias_pair_count']==244
assert prod['conditional_factory_unique_groups']==ind['conditional_factory_group_count']==264
assert prod['conditional_factory_unique_parameters']==ind['conditional_factory_unique_parameters']==34546498
assert all(t['all_finite'] for t in prod['tensor_records'])
producer_hashes={t['name']:t['raw_sha256'] for t in prod['tensor_records']}
assert set(producer_hashes)==set(ind['finite_checked_names'])
for pair in ind['alias_pairs']:
    a,b=pair['names']
    assert pair['bitwise_equal'] and pair['left_sha256']==producer_hashes[a] and pair['right_sha256']==producer_hashes[b]
assert len(ledger['attempts'])==2 and ledger['redirect_count']==1
assert ledger['observed_body_bytes']==sum(a['observed_body_bytes'] for a in ledger['attempts'])==272769016
assert ledger['observed_body_bytes']<=280*1024**2
assert ledger['automatic_retries']==0 and not ledger['signed_URLs_retained'] and not ledger['raw_redirect_error_bodies_retained']
assert all(a['body_complete'] and a['observed_body_bytes']==a['declared_body_bytes'] and a['elapsed_seconds']<=180 for a in ledger['attempts'])
assert [a['status'] for a in ledger['attempts']]==[302,206]
assert [a['hostname'] for a in ledger['attempts']]==['huggingface.co','us.aws.cdn.hf.co']
last=ledger['attempts'][-1]
assert last['observed_body_sha256']==ind['artifact']['sha256']
assert last['Content_Range']=='bytes 0-272767903/272767904'
assert last['ETag']=='"194bac772d9f80025066c6fc7e118a446b464074d29e234c22256674906133d9"'
assert last['Content_Encoding']=='identity'
assert controller['wall_seconds']<=240
assert ind['wall_seconds']<=180 and ind['cpu_seconds']<=120 and ind['max_rss_bytes']<=4*1024**3
assert prod['offline_wall_seconds']<=180 and prod['offline_cpu_seconds']<=120 and prod['max_rss_bytes']<=4*1024**3
report={'status':'PASS','independent_and_producer_named_finite_counts_match':True,
    'independent_alias_hashes_match_producer_raw_hashes':488,'independent_bytewise_pairs_match':244,
    'conditional_group_and_parameter_counts_match':True,'checker_source_unchanged':True,
    'live_attempts':2,'redirects':1,'observed_response_body_bytes':272769016,'retries':0,
    'full_artifact_bytes':272767904,'artifact_full_sha256':ind['artifact']['sha256'],
    'runtime_and_resource_limits_pass':True,'tensor_reads_by_this_comparison':0,'model_calls':0,'network_calls':0,
    'inputs':[pin(p) for p in [REVIEW/'verification-v1.json',RUN/'qualification.json',RUN/'ledger.json',RUN/'controller.json',ROOT/'CHECKPOINT_INDEPENDENT_GO.json']]}
out=REVIEW/'receipt-comparison.json'
with out.open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(pin(out)))
