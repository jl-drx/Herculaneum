"""Synthetic ties, undefined metrics, immutable support and saved calibration-barrier checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from comparison_common import (ROLES,CALIBRATION_IDS,sources,module,record,write_json,verify_locks_ready,
    equal_context_mean,require_primary_coverage,require,validate_ready_receipts,load_arrays)
from analyze_comparison import derived,macro


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False);source,_=sources()
    selector=module(source['cutoff'],'synthetic_retained_selector');scorer=module(source['metrics'],'synthetic_retained_metrics')
    checks=[]
    def passed(name):checks.append(name)
    def rejected(name,action):
        try:action()
        except (ValueError,KeyError):passed(name);return
        raise AssertionError('Expected rejection: '+name)
    for count in [19,20,21]:
        scores=np.zeros(count,dtype=np.float32);scores[:2]=1
        cut=selector.choose_development_threshold(scores)
        require(cut['maximum_false_positives']==count//20 and cut['threshold']>1
            and cut['development_false_positives']==0,'Crossing tie not conservatively excluded');passed('ties-N'+str(count))
    rejected('nonfinite-negative',lambda:selector.choose_development_threshold(np.array([np.nan],np.float32)))
    y=np.array([True,False,True,False]);s=np.array([.9,.9,.1,.1],np.float32)
    values,_=scorer.binary_metrics(y,s);require(values['roc_auc']==values['average_precision']==.5,'Mixed tie ranking differs');passed('mixed-score-tie')
    empty=derived(y,s,1.1,selector)
    require(empty['precision'] is None and empty['zero_predicted_positives'] and empty['f1']==0
        and empty['balanced_accuracy']==.5,'All-negative metrics differ');passed('all-negative-operating-point')
    require(equal_context_mean([None,.8]) is None and equal_context_mean([.2,.8])==.5,'Undefined macro dropped');passed('undefined-macro-preserved')
    rows=[dict(auc=.5,ap=.5,supplied_positive_prevalence=.5,operating={k:dict(empty) for k in ['retained_fourteenth','frag3_only']}) for _ in range(2)]
    rows[1]['operating']['frag3_only']['precision']=.9
    require(macro(rows)['operating']['frag3_only']['precision'] is None,'Undefined constituent dropped');passed('macro-precision-preserved')
    field=np.zeros((384,384),np.float32);support=np.zeros(field.shape,bool);support[100,100]=True;evaluated=np.ones(field.shape,bool)
    require_primary_coverage(field,evaluated,support);passed('supported-zero-retained')
    evaluated[100,100]=False;rejected('primary-unscored-rejected',lambda:require_primary_coverage(field,evaluated,support));evaluated[:]=True
    field[100,100]=np.nan;rejected('primary-nonfinite-rejected',lambda:require_primary_coverage(field,evaluated,support));field[:]=0
    field[0,0]=np.nan;require_primary_coverage(field,evaluated,support);passed('excluded-pixel-does-not-change-support')
    field[100,100]=1.1;rejected('primary-range-rejected',lambda:require_primary_coverage(field,evaluated,support))
    data=args.output_dir/'synthetic-input';data.mkdir();labels=np.zeros((384,384),np.uint8);labels[:,:192]=1
    border=np.zeros((384,384),bool);border[32:-32,32:-32]=True
    values=dict(surface=np.ones((17,384,384),np.float32),labels=labels,primary_mask=border,
        known_mask=np.ones((384,384),bool),surface_mask=np.ones((384,384),bool))
    for name,value in values.items():np.save(data/(name+'.npy'),value,allow_pickle=False)
    case=dict(arrays={k:record(data/(k+'.npy')) for k in values},primary_positive=51200,primary_negative=51200)
    load_arrays({'legacy_prepare':source['legacy_prepare']},case);passed('literal-binary-label-contract')
    labels[100,100]=2;np.save(data/'labels.npy',labels,allow_pickle=False);case['arrays']['labels']=record(data/'labels.npy')
    rejected('nonbinary-label-rejected',lambda:load_arrays({'legacy_prepare':source['legacy_prepare']},case))
    planpath=args.output_dir/'synthetic-plan.json';write_json(planpath,{'synthetic':True})
    plan={'_plan_record':record(planpath),'models':{role:{'state_sha256':role} for role in ROLES},
        'cases':[{'id':case,'primary_negative':20} for case in CALIBRATION_IDS]}
    ready={'status':'PASS','plan':plan['_plan_record'],'roles':ROLES,'models':{},'threshold_locks':{},
        'calibration_tile_forwards':150,'evaluation_tile_forwards':0}
    for role in ROLES:
        directory=args.output_dir/role;directory.mkdir();regional={case:selector.choose_development_threshold(np.zeros(20,np.float32)) for case in CALIBRATION_IDS}
        write_json(directory/'thresholds.json',dict(role=role,state_sha256=role,plan=plan['_plan_record'],regional=regional,
            primary=max(row['threshold'] for row in regional.values())))
        lock=record(directory/'thresholds.json');ready['threshold_locks'][role]=lock
        write_json(directory/'result.json',dict(status='PASS',role=role,phase='calibration',plan=plan['_plan_record'],
            state_sha256=role,tile_forward_count=50,threshold_lock=lock));ready['models'][role]=record(directory/'result.json')
    barrier=args.output_dir/'locks-ready.json';write_json(barrier,ready)
    require(list(json.loads(barrier.read_text())['threshold_locks'])!=ROLES,'Fixture failed to expose sorted-key order')
    verify_locks_ready(plan,barrier,record(barrier)['sha256']);passed('saved-sorted-JSON-barrier-roundtrip')
    bad=dict(ready,threshold_locks={k:v for k,v in ready['threshold_locks'].items() if k!=ROLES[0]})
    write_json(args.output_dir/'missing-lock.json',bad);rejected('missing-lock-rejected',lambda:verify_locks_ready(plan,args.output_dir/'missing-lock.json',record(args.output_dir/'missing-lock.json')['sha256']))
    rejected('tampered-barrier-hash-rejected',lambda:verify_locks_ready(plan,barrier,'0'*64))
    bad=dict(ready,evaluation_tile_forwards=1);write_json(args.output_dir/'premature-evaluation.json',bad)
    rejected('premature-evaluation-rejected',lambda:verify_locks_ready(plan,args.output_dir/'premature-evaluation.json',record(args.output_dir/'premature-evaluation.json')['sha256']))
    role=ROLES[0];corrupt=json.loads(Path(ready['threshold_locks'][role]['path']).read_text())
    corrupt['primary']=float('inf')
    for row in corrupt['regional'].values():row['threshold']=float('inf')
    corrupt_path=args.output_dir/'nonfinite-threshold.json';corrupt_path.write_text(json.dumps(corrupt)+'\n')
    receipt=json.loads(Path(ready['models'][role]['path']).read_text());receipt['threshold_lock']=record(corrupt_path)
    bad_receipt=args.output_dir/'nonfinite-result.json';write_json(bad_receipt,receipt)
    bad=json.loads(json.dumps(ready));bad['threshold_locks'][role]=record(corrupt_path);bad['models'][role]=record(bad_receipt)
    write_json(args.output_dir/'nonfinite-locks.json',bad)
    rejected('nonfinite-lock-rejected',lambda:verify_locks_ready(plan,args.output_dir/'nonfinite-locks.json',record(args.output_dir/'nonfinite-locks.json')['sha256']))
    descriptor=args.output_dir/'descriptor.json';freeze=args.output_dir/'freeze.json';design=args.output_dir/'design.json'
    for path in [descriptor,freeze,design]:write_json(path,dict(synthetic=True))
    go=args.output_dir/'source-go.json';ct=args.output_dir/'ct-pass.json'
    go_data=dict(status='GO',code_freeze=record(freeze),research_design=record(design))
    ct_data=dict(status='PASS',descriptor=record(descriptor),all_CT_arrays_exact=True)
    write_json(go,go_data);write_json(ct,ct_data)
    def ready_check():validate_ready_receipts(record(descriptor),record(freeze),record(design),record(go),record(ct))
    ready_check();passed('exact-source-GO-and-CT-PASS')
    for name,bad in [('empty-CT-receipt',{}),('wrong-CT-status',dict(ct_data,status='GO')),
        ('truthy-not-typed-exact',dict(ct_data,all_CT_arrays_exact=1)),('unrelated-descriptor',dict(ct_data,descriptor=record(design)))]:
        write_json(ct,bad);rejected(name,ready_check)
    write_json(ct,ct_data)
    for name,bad in [('wrong-source-status',dict(go_data,status='PASS')),('unrelated-source-freeze',dict(go_data,code_freeze=record(design)))]:
        write_json(go,bad);rejected(name,ready_check)
    write_json(go,go_data)
    receipt=dict(status='PASS',checks=checks,source_files=[record(p) for p in sorted(Path(__file__).parent.glob('*.py'))],
        actual_operations=dict(synthetic_arrays=True,model_calls=0,checkpoint_reads=0,real_input_array_reads=0,network_calls=0))
    write_json(args.output_dir/'fixture-receipt.json',receipt);print(json.dumps(dict(status='PASS',checks=len(checks),receipt=record(args.output_dir/'fixture-receipt.json'))))


if __name__=='__main__':main()
