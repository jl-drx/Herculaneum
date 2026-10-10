"""Bind exact retained source identities before evaluating the audit."""
import platform
import numpy as np
from audit_common import *
require(not (HERE/'GO.json').exists() and not (HERE/'run-v1').exists(),'New audit freeze required')
prior=json.loads((REPORTS/'thirtythird-experiment/PLAN.json').read_text())
p33=json.loads((REPORTS/'thirtythird-experiment/run-v1/COMPLETE.json').read_text())
source=json.loads((REPORTS/'thirtysecond-experiment/pixels-v2/COMPLETE.json').read_text())
pooled=json.loads((REPORTS/'thirtysecond-experiment/pooled-v1/COMPLETE.json').read_text())
contexts=[]
for c in prior['contexts']:
    name=c['id'];arrays=next(x['arrays'] for x in p33['contexts'] if x['id']==name)
    contexts.append(dict(id=name,raw=source['raw_arrays'][name],core64=c['core64'],prior_arrays=arrays,
        pooled=pooled['contexts'][name]['array'],source_bbox_yxhw=c['source_bbox_yxhw']))
plan=dict(created_at_utc=now(),contexts=contexts,features=['intensity','contrast'],polarities=[1,-1],supports=['original','interior'],
    scales=dict(single=list(range(1,64)),triplet=list(range(2,63,3))),limits=dict(worker_seconds=120,verification_seconds=120,peak_RSS_bytes=1024**3,new_phase_bytes=64*1024**2),
    max_network_requests=0,max_model_forwards=0,expected_metric_records=2688,expected_distribution_records=1344,
    prior_primary=pin(REPORTS/'thirtysecond-experiment/scores-v1/RESULTS.json'))
save(HERE/'PLAN.json',plan)
save(HERE/'TEST_RECEIPT.json',dict(status='PASS',command='python -B -m unittest -v test_features',passed=7,failed=0,errors=0,skipped=0))
save(HERE/'SETUP_RECEIPT.json',dict(status='PASS_AFTER_DIRECTORY_CREATION',initial_command_started=False,
    initial_error='Requested workdir did not exist; exec process creation rejected before shell command execution',
    correction='Created the new phase directory from /workspace, then wrote and tested helpers',scientific_run_attempts_at_freeze=0,
    prior_integrity=dict(status='PASS',record_count=20780,retained_bytes=7030164240,manifest_sha256='7e0a3d35f77f9802e1339b9621b434a9a9f84b6057d6b5fb19919e20ad25b3bb')))
files=[pin(HERE/p) for p in ['PROTOCOL.md','PLAN.json','audit_common.py','features.py','run_audit.py','test_features.py','TEST_RECEIPT.json','SETUP_RECEIPT.json','freeze_audit.py']]
files += [pin(REPORTS/p) for p in ['twentyninth-experiment/reference_metrics.py','thirtysecond-experiment/pixels-v2/COMPLETE.json','thirtysecond-experiment/pooled-v1/COMPLETE.json','thirtythird-experiment/PLAN.json','thirtythird-experiment/run-v1/COMPLETE.json']]
files += [plan['prior_primary']]
for c in contexts:
    for k in ['raw','core64','prior_arrays','pooled']:verify(c[k]);files.append(c[k])
save(HERE/'GO.json',dict(status='GO_FIXED_CACHED_CT_AUDIT',created_at_utc=now(),files=files,
    python=platform.python_version(),numpy=np.__version__,network_requests=0,model_forwards=0))
print(json.dumps(dict(status='GO',GO=pin(HERE/'GO.json'),contexts=len(contexts),metric_records=2688)))
