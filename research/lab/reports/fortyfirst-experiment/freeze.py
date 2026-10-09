"""Freeze exactly one supervision-gradient control before any model execution."""
import ast
from common import *
prior=json.loads((REPORTS/'fortieth-experiment/PLAN.json').read_text())
test=json.loads((HERE/'TEST_RECEIPT.json').read_text())
require(test['passed']==test['tests_run']==8 and test['failed']==test['errors']==test['skipped']==0,'All eight fixtures')
p={k:prior[k] for k in ['training_plan','historical_control32','retention_plan','paris_plan','pooled_shared','cases','metrics','cutoff','batch_arrays','data_audit','data_verification','control_training','control_evaluation']}
p.update(study='phase41-new-crop-supervision-gradient-control',arms=['E_no_new_BCE'],updates=512,max_updates=512,evaluation_updates=[512],development_endpoints=[512],fixed_risk_updates=UPDATES,schedule=prior['schedule40'],teacher_receipt=pin(REPORTS/'fortieth-experiment/run-v1/teacher/COMPLETE.json'),full_training_baseline=pin(REPORTS/'fortieth-experiment/run-v1/D_replication/COMPLETE.json'),prior_metrics=pin(REPORTS/'fortieth-experiment/METRICS.json'),reference_receipts={**prior['reference_receipts'],'D_replication-512':pin(REPORTS/'fortieth-experiment/run-v1/eval-D_replication-512/COMPLETE.json')},protocol=pin(HERE/'PROTOCOL.md'),limits=dict(wall_seconds=1800,worker_rss_bytes=4*1024**3,phase_bytes=2*1024**3),new_model_forwards_allowed=905,new_optimizer_updates_allowed=512,teacher_forwards=0,CPU_monetary_cap=None,GPU_pilot_total_cap_GBP=10,mechanism='three-way exact descriptive dominance against D512, conditional on fixed label-selected support',objective='original BCE / joint N + unchanged joint seed42 KL; full original objective diagnostic')
p['success']=p['gate']=dict(Paris_AP_gain_vs_seed42=.01,Paris_AP_gain_vs_A512=.01,regional_AP_AUROC_min_vs_seed42=-.02)
for f in HERE.glob('*.py'):ast.parse(f.read_text())
save(HERE/'PLAN.json',p)
files={r['path']:r for r in json.loads((REPORTS/'fortieth-experiment/GO.json').read_text())['files']}
def collect(v):
    if isinstance(v,dict):
        if {'path','sha256','size_bytes'}<=v.keys():
            r={k:v[k] for k in ['path','sha256','size_bytes']};verify(r);require(r['path'] not in files or files[r['path']]==r,'Pin collision');files[r['path']]=r
        else:
            for x in v.values():collect(x)
    elif isinstance(v,list):
        for x in v:collect(x)
collect(p)
for item in [p['teacher_receipt'],p['full_training_baseline'],*p['reference_receipts'].values()]:collect(json.loads(verify(item).read_text()))
for f in HERE.glob('*'):
    if f.is_file():collect(pin(f))
for r in files.values():verify(r)
save(HERE/'GO.json',dict(status='GO_APPROVED_CPU_SUPERVISION_CONTROL',created_at_utc=now(),user_authorization='User agrees with recommended new-crop supervised-loss control,9 October2026',files=[files[k] for k in sorted(files)],scientific_model_calls_before_GO=0,teacher_forwards=0,student_training_forwards=705,development_forwards=200,optimizer_updates=512,network_requests_during_model_execution=0,GPU_spend_GBP=0))
print(json.dumps(dict(status='GO',GO=pin(HERE/'GO.json'),files=len(files))))
