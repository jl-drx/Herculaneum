"""Prepare audit identities from JSON/byte hashes only; never decode array values."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
from audit_objective import ARMS,WINDOWS,record,verify,require

HERE=Path(__file__).resolve().parent
LAB=Path('/workspace/herculaneum-lab')
SOURCE_PLAN=dict(path=str(LAB/'reports/tenth-experiment/implementation/final-plan/control-plan.json'),
    size_bytes=109068,sha256='9e2c9f4f778c3f794e103191fd413937b521fa0d0b72cab0e3905778ce3e3a55')
SOURCE_RESULTS=dict(path=str(LAB/'reports/tenth-experiment/implementation/analysis/metrics.json'),
    size_bytes=57942,sha256='ff8e549c93ae1430373ff956a5448b5e10abf7202ede858828b0f8192362b751')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--review',type=Path,action='append',default=[]);parser.add_argument('--ready',action='store_true')
    args=parser.parse_args();base=json.loads(verify(SOURCE_PLAN).read_text());results=json.loads(verify(SOURCE_RESULTS).read_text())
    require(results['status']=='PASS' and results['plan']==SOURCE_PLAN,'Retained objective results differ')
    require(base['training']['loss_smoothing']==.5 and base['training']['manual_coefficient']==1.
        and base['training']['pseudo_coefficient']==.25 and base['training']['steps']==32
        and base['preservation']['coefficient']==1. and base['preservation']['temperature']==1.,'Source objective differs')
    ninth_plan=base['historical_plan_records']['A_smooth'];require(ninth_plan==base['historical_plan_records']['B_smooth'],'Historical plan differs')
    ninth=json.loads(verify(ninth_plan).read_text());regions=[];inputs={}
    for case in base['adaptation']:
        require(case['physical_group']=='PHerc0139','Adaptation specimen differs')
        arrays={key:case['inputs'][key] for key in ['inklabels','supervision_mask','validation_mask']}
        arrays.update(observed=case['inputs']['acquisition_observed_mask'],teacher43_scores=case['teacher_scores'],teacher43_evaluated=case['teacher_evaluated'])
        regions.append(dict(id=case['id'],arrays=arrays))
    for arm in ARMS:
        result_record=results['model_results'][arm];value=json.loads(verify(result_record).read_text());training=value['training']
        require(value['status']=='PASS' and value['arm']==arm and value['plan']==SOURCE_PLAN
            and training['smoothing_epsilon']==.5 and training['preservation_lambda']==1.,'Training receipt differs')
        inputs[arm]=dict(source_result=result_record,student_logits=training['preupdate_logits']['student'],
            teacher_logits=training['preupdate_logits']['teacher'],steps=training['steps_log'],
            ninth_steps=base['historical_step_logs'][arm.removesuffix('_preserve')+'_smooth'])
    source_code=[base['code']['run_worker.py'],base['code']['preservation.py'],base['code']['analyze_control.py'],
        ninth['code']['run_worker.py'],ninth['code']['smoothing.py']]
    mechanics_path=str(LAB/'reports/eighth-experiment/implementation/training_mechanics.py')
    helper_path=str(LAB/'reports/eighth-experiment/implementation/experiment_common.py')
    pins={item['path']:item for item in base['verified_inputs']}
    source_code.extend([pins[mechanics_path],pins[helper_path]])
    require(not args.ready or args.review,'Root reviews required before ready audit')
    plan=dict(kind='retained-online-objective-audit-v1',ready_for_execution=args.ready,arms=ARMS,windows=WINDOWS,
        source_plan=SOURCE_PLAN,source_results=SOURCE_RESULTS,ninth_source_plan=ninth_plan,source_code=source_code,
        schedule=base['schedule'],regions=regions,arm_inputs=inputs,
        code=record(HERE/'audit_objective.py'),preparer=record(Path(__file__).resolve()),
        design=record(HERE/'AUDIT_DESIGN.md'),reviews=[record(p) for p in args.review],
        hypothesis_source=record(LAB/'reports/eleventh-experiment/NEXT_STUDY_DRAFT.md'),
        objective_convention=dict(smoothing_epsilon=.5,manual_coefficient=1.,pseudo_coefficient_B=.25,
            preservation_lambda=1.,pseudo_teacher='retained seed43 blended score >=0.5',preservation_teacher='seed42 same-input patch logits',
            manual_support='observed & S & ~V',pseudo_support='observed & evaluated43 & finite43 & ~manual',
            R_coefficient=0.,hard_BCE_role='secondary only',ninth_role='historical scalar receipts only'),
        tolerances=dict(float32_scalar_absolute=2e-6,float64_identity_absolute=1e-12,KL_mean_floor=-1e-6,first_logit_absolute=1e-7),
        limits=dict(wall_seconds=60,cpu_seconds=120,rss_bytes=1024**3,address_space_bytes=8*1024**3,output_bytes=8*1024**2),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,threads=4),
        forbidden_operations=['network','checkpoint deserialization','model construction','model forward/backward','optimizer','new predictions'])
    records={}
    def collect(value):
        if isinstance(value,dict):
            if set(value)=={'path','size_bytes','sha256'}:
                verify(value);require(value['path'] not in records or records[value['path']]==value,'Conflicting source identities')
                records[value['path']]=value
            else:
                for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(plan);plan['verified_inputs']=list(records.values())
    args.output_dir.mkdir(parents=True,exist_ok=False);path=args.output_dir/'audit-plan.json'
    path.write_text(json.dumps(plan,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',plan=record(path),array_values_decoded=0,ready_for_execution=args.ready)))

if __name__=='__main__':main()
