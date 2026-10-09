"""Small frozen-panel gates, reusing unchanged sixth-stage file/model contracts."""
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

from lock_development_thresholds import load_sixth_common

base = load_sixth_common()
ARMS = base.ARMS
HERE = Path(__file__).resolve().parent
ROLES = base.INPUT_NAMES


def load_plan(path, execute=False):
    path = Path(path).resolve(strict=True)
    base.require(path.stat().st_size <= 1024*1024, 'Panel plan exceeds1MiB')
    plan = json.loads(path.read_text())
    base.require(plan['kind']=='fixed-reference-panel-v1' and plan['model_order']==ARMS, 'Unexpected panel/model order')
    if execute:
        base.require(plan['ready_for_execution'] is True, 'Panel not frozen ready')
    for item in plan['verified_inputs']:
        base.verify_record(item)
    for name in ['run_panel.py','analyze_panel.py','panel_common.py','operating_point.py','lock_development_thresholds.py']:
        base.require(base.verify_record(plan['code'][name]) == HERE/name, 'Panel execution source differs')
    metric_path=base.verify_record(plan['binary_metrics'])
    base.require(metric_path==Path('/workspace/herculaneum-lab/reports/sixth-experiment/benchmark/binary_metrics.py')
        and plan['binary_metrics']['sha256']=='8fe2848e0b66aa0c9dcd6a05e33db0c1380e4b9781a115262585ca551891923f', 'Metric source differs')
    audit=json.loads(base.verify_record(plan['model_contract_audit']).read_text())
    base.require(audit['status']=='passed' and audit['tensor_schema_identical_across_four_arms']
        and audit['matched_pair']['matched_settings'], 'Four-arm model audit not passed')
    base.verify_record(plan['model_contract_helper'])
    base.require([m['label'] for m in plan['models']]==ARMS, 'Four exact model arms required')
    for model in plan['models']:
        base.require(model['sha256']==audit['models'][model['label']]['sha256']
            and model['embedded_step']==74999, 'Model differs from audited artifact')
    lock=json.loads(base.verify_record(plan['threshold_lock']).read_text())
    base.require(lock['status']=='locked_development_thresholds_no_inference'
        and lock['development_case']=='pherc1667-w029' and lock['model_order']==ARMS, 'Development lock differs')
    base.require(lock['code']['operating_point']==plan['code']['operating_point.py'], 'Threshold helper changed after lock')
    for role in ARMS:
        threshold=lock['thresholds'][role]
        base.require(threshold['target_specificity_numerator']==95 and threshold['target_specificity_denominator']==100
            and threshold['maximum_false_positives']==threshold['negative_count']//20
            and threshold['development_false_positives']<=threshold['maximum_false_positives'], 'Threshold rule differs')
        base.require(lock['models_and_maps'][role]['checkpoint']['sha256']==audit['models'][role]['sha256'], 'Threshold model identity differs')
    base.require(plan['limits']['threads']==4 and plan['limits']['automatic_retries']==0
        and 0<plan['limits']['per_worker_wall_seconds']<=120
        and 0<plan['limits']['scheduled_tiles']<=400
        and 0<plan['limits']['maximum_output_bytes']<=128*1024*1024, 'Panel resource cap differs')
    cases=plan['cases']
    base.require(2<=len(cases)<=4 and len({c['id'] for c in cases})==len(cases), 'Require2–4 uniquely named reserved contexts')
    base.require(all(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',c['id']) for c in cases),'Unsafe context identifier')
    groups=[c['physical_group'] for c in cases]
    base.require(set(groups)=={'PHerc0139','PHerc0814'} and all(groups.count(g)<=2 for g in set(groups)), 'Require the two declared other-scroll groups, at most2 contexts each')
    scheduled=0
    for case in cases:
        data=case['data'];shape=data['shape']
        base.require(case['role']=='reserved_validation' and shape==[21,384,384],
            'This pooled-only panel requires exact uint8[21,384,384] contexts')
        base.require(data['model_layer_start']==2 and data['model_layer_stop']==19
            and data['label_plane']==10 and data['pixel_spacing_um']==9.596,
            'This panel requires the verified2:19/label10/9.596µm contract')
        bbox=data['bbox_source_xywh']
        base.require(isinstance(bbox,list) and len(bbox)==4 and all(isinstance(v,int) and not isinstance(v,bool) for v in bbox)
            and bbox[2:]==[384,384] and bbox[0]>=0 and bbox[1]>=0
            and bbox[0]%128==0 and bbox[1]%128==0,'Source bbox must be chunk-aligned384²')
        base.verify_record(case['provenance'])
        h,w=shape[1:]
        scheduled +=4*((h-128)//64+1+int((h-128)%64!=0))*((w-128)//64+1+int((w-128)%64!=0))
    base.require(scheduled<=plan['limits']['scheduled_tiles'], 'Scheduled tile budget exceeded')
    for index,left in enumerate(cases):
        for right in cases[index+1:]:
            if left['physical_group']==right['physical_group']:
                a,b=left['data']['bbox_source_xywh'],right['data']['bbox_source_xywh']
                base.require(a[0]+384<=b[0] or b[0]+384<=a[0] or a[1]+384<=b[1] or b[1]+384<=a[1],
                    'Same-source contexts overlap in flattened model input; stop')
    return plan,path,lock


def load_case(case):
    inputs=case['inputs'];base.require(set(inputs)==set(ROLES), 'Six separate source/annotation roles required')
    arrays={key:np.load(base.verify_record(value),mmap_mode='r',allow_pickle=False) for key,value in inputs.items()}
    data=case['data'];surface=arrays['surface'];shape=tuple(data['shape'][1:])
    base.require(surface.dtype==np.uint8 and list(surface.shape)==data['shape'], 'Source array shape/dtype differs')
    decoded={}
    for key in ['inklabels','supervision_mask','validation_mask']:
        value=arrays[key];foreground=data['binary_foreground'][key]
        base.require(value.dtype==np.uint8 and value.shape==shape and foreground in [1,255]
            and set(np.unique(value).tolist())<={0,foreground}, 'Invalid explicit annotation encoding')
        decoded[key]=value==foreground
    for key in ['acquisition_observed_mask','source_observed_nonzero_mask']:
        base.require(arrays[key].dtype==np.bool_ and arrays[key].shape==shape, 'Distinct Boolean source support required')
    base.require(arrays['acquisition_observed_mask'].all(), 'Missing source support; no fill context allowed')
    occupancy=np.any(surface[data['model_layer_start']:data['model_layer_stop']]!=0,axis=0)
    base.require(np.array_equal(occupancy,arrays['source_observed_nonzero_mask']), 'Selected-window occupancy differs')
    labels=decoded['inklabels'];secondary=decoded['validation_mask'] & arrays['acquisition_observed_mask']
    margin=np.zeros(shape,dtype=bool);margin[64:shape[0]-64,64:shape[1]-64]=True
    primary=secondary & margin;counts=base.class_counts(labels,primary)
    base.require(base.sufficient(counts), 'STOP: fixed primary lacks>=256 supplied positives and negatives')
    summary=dict(primary=counts,secondary=base.class_counts(labels,secondary),
        raw_supervision_validation_overlap_pixels=int((decoded['supervision_mask']&decoded['validation_mask']).sum()),
        raw_supervision_pixels=int(decoded['supervision_mask'].sum()),
        observed_zero_profile_primary_pixels=int((primary&~occupancy).sum()),
        domain='Validation & actual acquisition support & fixed64margin; no occupancy/supervision exclusion or authenticated geometry claim.')
    return arrays,labels,primary,secondary,summary


def metric_module(plan):
    path=base.verify_record(plan['binary_metrics'])
    spec=importlib.util.spec_from_file_location('verified_sixth_binary_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
