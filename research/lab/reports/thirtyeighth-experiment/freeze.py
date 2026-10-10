"""Bind the authorized study after admission and mechanics checks, before model calls."""
import copy,ast
from common import *
prior=REPORTS/'thirtyseventh-experiment';audit=json.loads((HERE/'DATA_AUDIT.json').read_text());dv=json.loads((HERE/'DATA_VERIFICATION.json').read_text());test=json.loads((HERE/'TEST_RECEIPT.json').read_text())
require(audit['status']=='PASS_OPERATIONAL_PUBLIC_REFERENCE_SUPERVISION' and dv['status']=='PASS','Data gate')
require(test['passed']==test['tests_run']==7 and test['failed']==test['errors']==test['skipped']==0,'Mechanics gate')
p=copy.deepcopy(json.loads((prior/'PLAN.json').read_text()));p['study']='phase38-class-exposure-matched-training-coverage';p['batch_arrays']={kind:{name:pin(HERE/f'data/{kind}-{name}.npy') for name in ['inputs','mask','target']} for kind in ['mixed','original']}
p.update(data_audit=pin(HERE/'DATA_AUDIT.json'),data_verification=pin(HERE/'DATA_VERIFICATION.json'),schedule38=pin(HERE/'data/schedule.json'),control_training=pin(prior/'run-v1/A_control/COMPLETE.json'),control_evaluation=pin(prior/'run-v1/eval-A_control-512/COMPLETE.json'),prior_metrics=pin(prior/'METRICS.json'),protocol=pin(HERE/'PROTOCOL.md'),gate=dict(Paris_AP_gain_vs_seed42=.01,Paris_AP_gain_vs_A512=.01,regional_AP_AUROC_min_vs_seed42=-.02),max_updates=512,development_endpoints=[512])
save(HERE/'PLAN.json',p)
files={r['path']:r for r in json.loads((prior/'GO.json').read_text())['files']}
def add(r):
    verify(r);require(r['path'] not in files or files[r['path']]==r,'No pin collision');files[r['path']]=r
def collect(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256','size_bytes'}:add(value)
        else:
            for v in value.values():collect(v)
    elif isinstance(value,list):
        for v in value:collect(v)
collect(p);collect(audit);collect(json.loads((prior/'run-v1/eval-A_control-512/COMPLETE.json').read_text()))
for f in HERE.rglob('*'):
    if f.is_file() and '__pycache__' not in f.parts:add(pin(f))
for f in HERE.glob('*.py'):ast.parse(f.read_text())
save(HERE/'GO.json',dict(status='GO_APPROVED_CPU_TRAINING_COVERAGE',created_at_utc=now(),user_authorization='Proceed after phase37 training-coverage recommendation',files=[files[k] for k in sorted(files)],scientific_model_calls_before_GO=0,teacher_forwards=32,student_training_forwards=705,development_forwards=200,optimizer_updates=512,network_requests=0,GPU_spend_GBP=0))
print(json.dumps(dict(status='GO',files=len(files),GO=pin(HERE/'GO.json'))))
