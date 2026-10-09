"""Bind retained sources, fixed schedule and root-reviewed fresh descriptor."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment_common import HERE, record, verify, write_json, require, training_region
from training_mechanics import build_schedule

LAB = HERE.parents[2]


def retained_case(case, role):
    result = dict(id=case['id'], physical_group=case['physical_group'], inputs=dict(case['inputs']),
        data=case['data'], provenance=case['provenance'], evaluation_participation_role=role)
    if role == 'public_validation': result['inputs']['evaluation_participation_mask'] = result['inputs']['validation_mask']
    return result


def old_maps(case, base, label):
    directory = base / label
    meta = json.loads((directory/'metadata.json').read_text())
    require(meta['model']['label'] == label and meta['model']['sha256']=={'seed42':'e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab',
        'seed43':'2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f'}[label], 'Retained score model identity differs')
    require(record(directory/'scores.npy') == meta['raw_scores'] and record(directory/'evaluated-mask.npy') == meta['evaluated_mask'], 'Retained map receipt differs')
    return dict(scores=record(directory/'scores.npy'), evaluated=record(directory/'evaluated-mask.npy'),
        metadata=record(directory/'metadata.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-descriptor',type=Path,required=True)
    parser.add_argument('--data-review',type=Path,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--ready',action='store_true',help='Root-only after independent data/code review')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    p6=LAB/'reports/sixth-experiment/benchmark/comparison-plan.json'; p7=LAB/'reports/seventh-experiment/benchmark/panel-plan.json'
    six=json.loads(p6.read_text()); seven=json.loads(p7.read_text())
    adaptation=[retained_case(c,'retired_validation_adaptation') for c in seven['cases'][:2]]
    for case in adaptation:
        maps=old_maps(case,p7.parent/'run'/case['id'],'seed43')
        case.update(teacher_scores=maps['scores'],teacher_evaluated=maps['evaluated'],teacher_metadata=maps['metadata'])
    development=[dict(id='pherc1667-w029',physical_group='PHerc1667',inputs=dict(six['inputs']),
        data=six['data'],provenance=six['data_contract'],evaluation_participation_role='public_validation')]
    development[0]['inputs']['evaluation_participation_mask']=six['inputs']['validation_mask']
    development[0]['baseline_maps']=old_maps(development[0],p6.parent/'run','seed42')
    for c in seven['cases'][2:]:
        case=retained_case(c,'public_validation'); case['baseline_maps']=old_maps(case,p7.parent/'run'/case['id'],'seed42'); development.append(case)
    descriptor=json.loads(args.fresh_descriptor.read_text()); fresh=[]
    require(descriptor['status']=='fresh_reference_complete_no_inference','Fresh acquisition incomplete')
    source_plan=record(descriptor['source_plan_record']['path'])
    require(source_plan['sha256']==descriptor['source_plan_record']['sha256'] and source_plan['size_bytes']==descriptor['source_plan_record']['byte_length'],'Source-plan identity differs')
    for c in descriptor['cases']:
        require(c['physical_group']=='PHerc0814' and c['evaluation_participation_role']=='supplied_supervision'
            and c['shape_zyx']==[21,384,384] and c['model_layer_slice']==[2,19]
            and c['annotation_center_layer']==10 and c['pixel_spacing_um']==9.596,'Fresh contract differs')
        for item in [c['provenance_record'],*c['arrays'].values()]:
            actual=record(item['path']); require(actual['sha256']==item['sha256'] and actual['size_bytes']==item['byte_length'],'Fresh descriptor hash differs')
        fresh.append(dict(id=c['id'],physical_group=c['physical_group'],inputs={k:record(v['path']) for k,v in c['arrays'].items()
            if k in ['surface','inklabels','supervision_mask','validation_mask','acquisition_observed_mask',
                'source_observed_nonzero_mask','evaluation_participation_mask']},data=c,
            provenance=record(c['provenance_record']['path']),evaluation_participation_role='supplied_supervision'))
    require(len(fresh)==2,'Exactly two frozen fresh cases required')
    tiles=[]; exposure=[]
    for case in adaptation:
        _,m,target,p,pt=training_region(case); eligible=[]; union=np.zeros(m.shape,bool)
        for y in range(0,257,64):
            for x in range(0,257,64):
                local=m[y:y+128,x:x+128]; t=target[y:y+128,x:x+128]
                if int((local&(t==1)).sum())>=256 and int((local&(t==0)).sum())>=256:
                    eligible.append([x,y,128,128]); union[y:y+128,x:x+128]=True
        require(len(eligible)==6,'Declared six eligible tiles per region changed'); tiles.append(eligible)
        exposure.append(dict(id=case['id'],eligible_tiles=eligible,unique_manual_pixels=int((m&union).sum()),
            unique_pseudo_positive_pixels=int((p&union&(pt==1)).sum()),unique_pseudo_negative_pixels=int((p&union&(pt==0)).sum()),
            whole_context_pseudo_positive_pixels=int((p&(pt==1)).sum())))
    schedule_path=args.output_dir/'schedule.json'; write_json(schedule_path,build_schedule(tiles))
    plan=dict(kind='matched-continuation-v1',ready_for_execution=args.ready,
        initializer=next(c for c in six['models'] if c['label']=='seed42'),teacher=next(c for c in six['models'] if c['label']=='seed43'),
        helper=six['model_contract_helper'],metrics=seven['binary_metrics'],cutoff=record(LAB/'reports/seventh-experiment/benchmark/operating_point.py'),
        adaptation=adaptation,development=development,fresh=fresh,schedule=record(schedule_path),exposure=exposure,
        training=dict(steps=32,batch_size=2,seed=20261008,learning_rate=.0001,optimizer='AdamW',betas=[.9,.999],
            eps=1e-8,weight_decay=0,scheduler=None,gradient_clip_norm=1.,teacher_threshold=.5,loss_smoothing=0,
            manual_coefficient=1.,pseudo_coefficient=.25,balanced_manual_mixture=.5),
        limits=dict(wall_seconds=120,cpu_seconds=360,threads=4,rss_acceptance_bytes=4*1024**3,
            address_space_bytes=12*1024**3,output_bytes=700*1024**2,retries=0),
        calibration=dict(primary='max of three per-region conservative 95% development-negative cutoffs',secondary='pherc1667-w029 only',
            specificity_numerator=95,specificity_denominator=100),
        gate=dict(comparators=['B','A','frozen_seed42'],specificity_min=.95,mean_sensitivity_gain_min=.02,
            per_context_sensitivity_delta_min=-.02,per_context_auc_delta_min=-.01,per_context_ap_delta_min=-.02,
            promising_requires_all_comparisons=True,statistical_significance_claim=False),
        code={p.name:record(p) for p in sorted(HERE.glob('*.py'))})
    inputs=[record(p6),record(p7),record(args.fresh_descriptor),source_plan,*[record(p) for p in args.data_review],record(HERE/'code-freeze.json'),
        record(HERE.parent/'PROTOCOL.md'),record(HERE.parent/'IMPLEMENTATION_DECISIONS.md'),
        record(LAB/'run_ink_reference.py'),record(LAB/'vendor/villa-ink-baseline/source-manifest.json')]
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256','size_bytes'}<=value.keys():
                expected={k:value[k] for k in ['path','sha256','size_bytes']}; verify(expected); inputs.append(expected)
            for child in value.values(): collect(child)
        elif isinstance(value,list):
            for child in value: collect(child)
    collect(plan); plan['verified_inputs']=sorted({item['path']:item for item in inputs}.values(),key=lambda i:i['path'])
    write_json(args.output_dir/'training-plan.json',plan)
    print(json.dumps(dict(plan=record(args.output_dir/'training-plan.json'),schedule=record(schedule_path),exposure=exposure)))


if __name__=='__main__': main()
