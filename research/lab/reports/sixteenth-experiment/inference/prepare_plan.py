"""Prepare a draft or root-reviewed ready plan; never construct models or decode arrays."""
import argparse
import json
from pathlib import Path

from comparison_common import (HERE,ROLES,CALIBRATION_IDS,EVALUATION_IDS,PREPROCESSING,LIMITS,INFERENCE,
    sources,record,identity,read_json,write_json,require,collect,load_plan,DESIGN_SHA,RUNTIME_SHA,validate_ready_receipts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor',type=Path);parser.add_argument('--research-design',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--review',type=Path,action='append',default=[])
    parser.add_argument('--source-go',type=Path);parser.add_argument('--ct-verification',type=Path)
    parser.add_argument('--ready',action='store_true');args=parser.parse_args()
    source,pin=sources();benchmark=read_json(source['benchmark_contract'])
    require(benchmark['accuracy_measured'] is False and benchmark['CT_ready'] is False,'Label-only stage changed')
    selected={row['case_id']:row for row in benchmark['selected_contexts']};descriptor_record=None;descriptor=None
    if args.descriptor:
        descriptor_record=record(args.descriptor);descriptor=read_json(descriptor_record)
        require(descriptor['status']=='pooled_reference_complete_no_inference','CT inputs incomplete')
        require(all(descriptor['preprocessing'].get(k)==v for k,v in PREPROCESSING.items()),'CT pooling contract differs')
        require({c['id'] for c in descriptor['cases']}==set(selected),'Input contexts differ')
        inputs={c['id']:c for c in descriptor['cases']}
    cases=[]
    names=dict(labels='labels.npy',primary_mask='primary-mask.npy',known_mask='known-mask.npy',surface_mask='surface-mask.npy')
    for case_id in CALIBRATION_IDS+EVALUATION_IDS:
        row=selected[case_id];arrays={key:identity(row['files'][name]) for key,name in names.items()}
        arrays['surface']=None
        if descriptor:
            data=inputs[case_id]
            require(data['native_bbox_yxhw']==row['native_bbox_yxhw'],'CT source origin differs')
            require(set(data['arrays'])==set(arrays),'Input array roles differ')
            for key in names:require(identity(data['arrays'][key])==arrays[key],'Literal fifteenth mask/label changed')
            arrays['surface']=identity(data['arrays']['surface'])
        cases.append(dict(id=case_id,role=row['role'],fragment=row['fragment'],parent_papyrus=row['parent_papyrus'],
            native_bbox_yxhw=row['native_bbox_yxhw'],bbox_source_xywh=row['bbox_source_xywh'],
            primary_positive=row['primary_positive'],primary_negative=row['primary_negative'],arrays=arrays))
    design=record(args.research_design) if args.research_design else None
    if design:require(design['sha256']==DESIGN_SHA,'Root scientific design differs')
    runtime=record(HERE/'RUNTIME_PINS.json');require(runtime['sha256']==RUNTIME_SHA,'Runtime pin changed')
    freeze=record(HERE/'code-freeze.json') if (HERE/'code-freeze.json').exists() else None
    source_go=record(args.source_go) if args.source_go else None
    ct_verification=record(args.ct_verification) if args.ct_verification else None
    if args.ready:validate_ready_receipts(descriptor_record,freeze,design,source_go,ct_verification)
    plan=dict(kind='Paris-fragments-fixed-state-calibration-transfer-v1',ready_for_execution=args.ready,
        roles=ROLES,cases=cases,source_pins=pin,descriptor=descriptor_record,preprocessing=PREPROCESSING,
        inference=INFERENCE,limits=LIMITS,models=source['models'],research_design=design,runtime=runtime,
        reviews=[record(x) for x in args.review]+[x for x in [source_go,ct_verification] if x is not None],
        source_go=source_go,ct_verification=ct_verification,code_freeze=freeze,code={p.name:record(p) for p in sorted(HERE.glob('*.py'))},
        cutoff_contract=dict(original_rule='retained fourteenth development cutoffs, own primary per frozen model',
            new_rule='max of two Frag3 negative-only conservative95% specificity cutoffs',
            selector='unchanged seventh operating_point.py',operator='float64(score) >= threshold',
            fitting_support='literal fifteenth primary_mask & labels==0',evaluation_fitting=False),
        execution_order='all3 Frag3 workers -> all3 cutoff locks-ready -> all3 Frag1/2 workers -> analysis',
        planned_model_load_operations=dict(final_role_worker_constructions=6,initializer_strict_loads=6,
            candidate_raw_state_strict_loads=4,total_load_state_dict_calls=10),
        reporting='percontext and equalcontext perfragment; Paris2 equalfragment mean; no pixel pooling/CI/pvalue/promotion',
        complete_output_rule='Retain all3states x6fixedcontexts; stop on failure, no retry or substitution',
        **{key:source[key] for key in ['initializer','helper','shared','legacy_prepare','legacy_worker','cutoff','metrics','transitive_runtime_inputs','load_accounting_clarification']})
    if args.ready:
        records={};collect(plan,records);collect(source,records);collect(descriptor,records);plan['verified_inputs']=list(records.values())
    else:plan['verified_inputs']=[]
    args.output_dir.mkdir(parents=True,exist_ok=False);path=args.output_dir/'comparison-plan.json';write_json(path,plan)
    load_plan(path,execute=args.ready);print(json.dumps(dict(status='PASS',plan=record(path),model_calls=0,array_decodes=0)))


if __name__=='__main__':main()
