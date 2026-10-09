"""Bind inputs without loading or combining their scores."""
import sys
from common import *

require(not (HERE/'GO.json').exists() and not (HERE/'run-v1').exists(), 'New frozen comparison')
prior = LAB/'reports/thirtyeighth-experiment'
p = json.loads((prior/'PLAN-V2.json').read_text())
old = json.loads(verify(p['control_evaluation']).read_text())
coverage_path = prior/'run-v1/eval-C_coverage-512/COMPLETE.json'
coverage = json.loads(coverage_path.read_text())
cases = []
for c in p['cases']:
    if c['family']=='paris17':
        inputs = {k:c['arrays'][k] for k in ('labels','known_mask','primary_mask')}
        interior = [32,352]
    else:
        src = c['legacy_case']['inputs']
        inputs = dict(labels=src['inklabels'], observed=src['acquisition_observed_mask'], supervision=src['supervision_mask'], primary_mask=c['primary_mask'])
        interior = [64,320]
    sources = {'seed42':{k:c['baseline_metadata'][k] for k in ('scores','evaluated')}}
    for role,receipt in [('A_control-512',old),('C_coverage-512',coverage)]:
        row = next(r for r in receipt['contexts'] if r['id']==c['id'])
        sources[role] = {k:row[k] for k in ('scores','evaluated')}
    cases.append(dict(id=c['id'],parent=c['parent'],family=c['family'],positive=c['positive'],negative=c['negative'],interior=interior,inputs=inputs,sources=sources))
plan = dict(schema_version=1, study='phase39-fixed-midpoint', created_at_utc=now(), cases=cases, coefficient=.5, score_arithmetic='float64(.5)*float64(seed42_FP32) + float64(.5)*float64(A512_FP32)', score_output_dtype='float64', shape=[384,384], thresholds=[.5,p['cutoff']], metrics=p['metrics'], prior_metrics=pin(prior/'METRICS.json'), reference_plan=pin(prior/'PLAN-V2.json'), reference_receipts=[p['control_evaluation'],pin(coverage_path)], proposal=pin(prior/'NEXT_STUDY_DRAFT.md'), protocol=pin(HERE/'PROTOCOL.md'), limits=dict(seconds_per_process=120,peak_RSS_bytes=1024**3,phase_bytes=64*1024**2), advancement=dict(Paris_AP_gain_each=.01,regional_delta_minimum=-.02), descriptive='AP >= coverage512 and minimum AP/AUROC deltas >= corresponding coverage512 minima', all_regions_reused_development=True, new_model_forwards=0, optimizer_updates=0, acquisition_requests=0, GPU_spend_GBP=0, first_GPU_total_cap_GBP=10, CPU_monetary_cap=None, python=sys.version, numpy=np.__version__)
save(HERE/'PLAN.json',plan)
pins={}
def collect(obj):
    if isinstance(obj,dict):
        if {'path','sha256','size_bytes'}<=obj.keys():
            verify(obj); pins[obj['path']]={k:obj[k] for k in ('path','sha256','size_bytes')}
        else:
            for v in obj.values(): collect(v)
    elif isinstance(obj,list):
        for v in obj: collect(v)
collect(plan)
for path in [*HERE.glob('*.py'),HERE/'PROTOCOL.md',HERE/'PLAN.json',HERE/'TEST_RECEIPT.json']:
    item=pin(path);pins[item['path']]=item
save(HERE/'GO.json',dict(status='FROZEN_BEFORE_SCORES',created_at_utc=now(),pins=[pins[k] for k in sorted(pins)], post_result_relative_to_phase38=True))
print(json.dumps(dict(status='FROZEN_BEFORE_SCORES',GO=pin(HERE/'GO.json'),pins=len(pins))))
