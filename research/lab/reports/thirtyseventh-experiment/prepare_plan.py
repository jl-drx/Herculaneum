"""Bind retained metadata and approved design before any scientific model call."""
from study_common import *
require(not (HERE/'GO.json').exists(),'New freeze required')
train=REPORTS/'thirteenth-experiment/implementation/final-plan/training-plan.json'
paris=REPORTS/'twentieth-experiment/implementation/final-plan/experiment-plan.json'
ret=REPORTS/'thirteenth-experiment/retention/final-plan/retention-plan.json'
a=json.loads(train.read_text());b=json.loads(paris.read_text());d=json.loads(ret.read_text());cases=[]
for c in b['cases']:
    metadata=json.loads(verify(c['baseline_metadata']).read_text())
    cases.append(dict(id=c['id'],family='paris17',parent=c['parent_papyrus'],arrays=c['arrays'],layer_start=0,positive=c['primary_positive'],negative=c['primary_negative'],baseline_metadata=metadata,baseline_source=c['baseline_metadata'],baseline_result=c['baseline_result']))
for c in d['cases']:
    if c['family']=='pooled21':
        cases.append(dict(id=c['id'],family='pooled21',parent='PHerc0814',legacy_case=c['legacy_case'],primary_mask=c['primary_mask'],layer_start=2,positive=c['canonical_counts']['positive'],negative=c['canonical_counts']['negative'],baseline_metadata=c['historical_maps']['frozen_seed42']))
require(len(cases)==8 and sum(c['parent']=='Paris2' for c in cases)==4,'Exact development panel')
p=dict(training_plan=pin(train),paris_plan=pin(paris),retention_plan=pin(ret),historical_control32=pin(REPORTS/'thirteenth-experiment/implementation/run/A_preserve/result.json'),cases=cases,
    pooled_shared=d['shared'],metrics=d['metrics'],cutoff=.7558947205543518,arms=ARMS,updates=512,fixed_risk_updates=UPDATES,evaluation_updates=[32,512],
    success=dict(Paris_AP_gain=.01,per_region_AP_AUC_delta_min=-.02),limits=dict(wall_seconds=1800,worker_rss_bytes=4*1024**3,phase_bytes=2*1024**3),new_model_forwards_allowed=2146,new_optimizer_updates_allowed=1024)
save(HERE/'PLAN.json',p)
save(HERE/'TEST_RECEIPT.json',dict(status='PASS',passed=7,failed=0,errors=0,skipped=0,command='python -B -m unittest -v test_objectives.py'))
records={}
def collect(v):
    if isinstance(v,dict):
        if 'path' in v and 'sha256' in v:
            sizes=[v[k] for k in ['size_bytes','byte_length','bytes'] if k in v]
            require(sizes and len(set(sizes))==1,'Canonical size identity')
            q=dict(path=str(Path(v['path']).resolve()),size_bytes=sizes[0],sha256=v['sha256']);verify(q)
            require(q['path'] not in records or records[q['path']]==q,'Conflicting identity');records[q['path']]=q
        else:
            for vv in v.values():collect(vv)
    elif isinstance(v,list):
        for vv in v:collect(vv)
collect(p);collect(a['verified_inputs']);collect(a['shared_sources']);collect(a['code']);collect(a['historical']);collect(a['helper']);collect(a['first_step_probe']);collect(a['adaptation']);collect(a['schedule']);collect(b['source_code']['run_ink_reference.py']);collect(b['source_code']['normalization.py'])
for f in HERE.glob('*.py'):collect(pin(f))
for f in ['PROTOCOL.md','PLAN.json','TEST_RECEIPT.json']:collect(pin(HERE/f))
for f in ['thirteenth-experiment/implementation/run_worker.py','ninth-experiment/source-audit/upstream-nnunet-losses.py']:collect(pin(REPORTS/f))
save(HERE/'GO.json',dict(status='GO_APPROVED_CPU_BUDGET_OBJECTIVE',created_at_utc=now(),files=[records[k] for k in sorted(records)],new_network_requests=0,GPU_spend_GBP=0))
print(json.dumps(dict(GO=pin(HERE/'GO.json'),pinned_files=len(records),cases=[c['id'] for c in cases])))
