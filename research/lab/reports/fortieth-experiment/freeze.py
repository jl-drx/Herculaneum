"""Bind the crop replication after admission and fixtures, before any model call."""
import ast
from common import *
prior=json.loads((REPORTS/'thirtyeighth-experiment/PLAN-V2.json').read_text())
audit=json.loads((HERE/'DATA_AUDIT.json').read_text());data_check=json.loads((HERE/'DATA_VERIFICATION.json').read_text());test=json.loads((HERE/'TEST_RECEIPT.json').read_text())
require(audit['status']=='PASS_OPERATIONAL_PUBLIC_REFERENCE_SUPERVISION' and data_check['status']=='PASS','Data gate')
require(test['passed']==test['tests_run']==7 and test['failed']==test['errors']==test['skipped']==0,'Fixture gate')
# Explicit fields prevent obsolete arm/endpoint/success metadata leaking from a prior plan.
p={k:prior[k] for k in ['training_plan','historical_control32','retention_plan','paris_plan','pooled_shared','cases','metrics','cutoff']}
p.update(study='phase40-disjoint-source-crop-replication',arms=['D_replication'],updates=512,max_updates=512,evaluation_updates=[512],development_endpoints=[512],fixed_risk_updates=[32,128,256,384,512],batch_arrays={kind:{name:pin(HERE/f'data/{kind}-{name}.npy') for name in ['inputs','mask','target']} for kind in ['mixed','original']},data_audit=pin(HERE/'DATA_AUDIT.json'),data_verification=pin(HERE/'DATA_VERIFICATION.json'),schedule40=pin(HERE/'data/schedule.json'),control_training=prior['control_training'],control_evaluation=prior['control_evaluation'],prior_metrics=pin(REPORTS/'thirtyninth-experiment/run-v1/METRICS.json'),reference_receipts={'A_control-512':prior['control_evaluation'],'C_coverage-512':pin(REPORTS/'thirtyeighth-experiment/run-v1/eval-C_coverage-512/COMPLETE.json'),'midpoint':pin(REPORTS/'thirtyninth-experiment/run-v1/COMPLETE.json')},protocol=pin(HERE/'PROTOCOL.md'),limits=dict(wall_seconds=1800,worker_rss_bytes=4*1024**3,phase_bytes=2*1024**3),new_model_forwards_allowed=937,new_optimizer_updates_allowed=512,descriptive_replication='Paris AP and minimum regional AP/AUROC deltas against seed42 each >= fixed midpoint',CPU_monetary_cap=None,GPU_pilot_total_cap_GBP=10)
p['gate']=p['success']=dict(Paris_AP_gain_vs_seed42=.01,Paris_AP_gain_vs_A512=.01,regional_AP_AUROC_min_vs_seed42=-.02)
if (HERE/'PLAN.json').exists():require(json.loads((HERE/'PLAN.json').read_text())==p,'Preflight plan content unchanged')
else:save(HERE/'PLAN.json',p)
files={r['path']:r for r in json.loads((REPORTS/'thirtyseventh-experiment/GO.json').read_text())['files']}
def add(r):
    verify(r);require(r['path'] not in files or files[r['path']]==r,'Pin collision');files[r['path']]=r
def collect(v):
    if isinstance(v,dict):
        if {'path','sha256','size_bytes'}<=v.keys():add({k:v[k] for k in ['path','sha256','size_bytes']})
        else:
            for x in v.values():collect(x)
    elif isinstance(v,list):
        for x in v:collect(x)
collect(p)
for r in p['reference_receipts'].values():collect(json.loads(verify(r).read_text()))
for f in HERE.rglob('*'):
    if f.is_file() and '__pycache__' not in f.parts:add(pin(f))
for f in HERE.glob('*.py'):ast.parse(f.read_text())
save(HERE/'GO.json',dict(status='GO_APPROVED_CPU_CROP_REPLICATION',created_at_utc=now(),user_authorization='Proceed as recommended after phase39',files=[files[k] for k in sorted(files)],scientific_model_calls_before_GO=0,teacher_forwards=32,student_training_forwards=705,development_forwards=200,optimizer_updates=512,network_requests_during_model_execution=0,GPU_spend_GBP=0,admission_stages=2))
print(json.dumps(dict(status='GO',GO=pin(HERE/'GO.json'),files=len(files))))
