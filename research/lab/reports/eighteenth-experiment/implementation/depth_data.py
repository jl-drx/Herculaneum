"""Scientific IO is explicit and called only after the frozen root GO is checked."""
import hashlib
import json
import numpy as np
import torch

from depth_common import require, verify, read_json, module, state_hash
from depth_adapter import features


def load_array(pin, shape=None, dtype=None):
    array = np.load(verify(pin), allow_pickle=False, mmap_mode='r')
    require((shape is None or array.shape == shape) and (dtype is None or array.dtype == dtype),
            'Array shape/dtype differs: ' + pin['path'])
    require(np.isfinite(array).all(), 'Nonfinite scientific array: ' + pin['path'])
    return array


def runtime(plan):
    reference = module(plan['source_code']['run_ink_reference.py'], 'phase18_reference')
    normalizer = module(plan['source_code']['normalization.py'], 'phase18_normalization')
    return dict(reference=reference, normalize_robust=normalizer.normalize_robust)


def training_batches(plan, arm, context):
    base = read_json(plan['training_base_plan'])
    require([c['id'] for c in base['adaptation']] == ['pherc0139-w016-rank01', 'pherc0139-w016-rank02']
            and all(c['physical_group'] == 'PHerc0139' for c in base['adaptation']), 'Training specimen scope changed')
    require(base['schedule'] == plan['schedule'] and base['historical_initial_state_sha256'] == plan['backbone_state_sha256'],
            'Schedule/backbone lineage differs')
    require(base['initializer']==plan['backbone_checkpoint_lineage'], 'Backbone checkpoint lineage differs')
    for arm_name in ['A_preserve','B_preserve']:
        require(base['historical'][arm_name]['teacher']==plan['teacher_caches'][arm_name]
                and base['historical'][arm_name]['steps']==plan['historical_step_logs'][arm_name]
                and base['historical'][arm_name]['result']==plan['historical_training_results'][arm_name],
                'Cached training lineage differs')
        historical=read_json(plan['historical_training_results'][arm_name])
        require(historical['status']=='PASS' and historical['arm']==arm_name
                and historical['initial_state_sha256']==plan['backbone_state_sha256']
                and historical['training']['preupdate_logits']['teacher']==plan['teacher_caches'][arm_name]
                and historical['training']['steps_log']==plan['historical_step_logs'][arm_name],
                'Historical training result source differs')
    shared = module(plan['source_code']['experiment_common.py'], 'phase18_historical_io')
    mechanics = module(plan['source_code']['training_mechanics.py'], 'phase18_historical_augmentation')
    regions = [shared.training_region(case) for case in base['adaptation']]
    schedule = read_json(plan['schedule'])
    histories = {a:[json.loads(line) for line in verify(pin).read_text().splitlines()]
                 for a,pin in plan['historical_step_logs'].items()}
    teachers = [load_array(pin, (32,2,1,128,128), np.float32) for pin in plan['teacher_caches'].values()]
    require(len(teachers) == 2 and np.array_equal(teachers[0].view(np.uint32), teachers[1].view(np.uint32)),
            'Matched cached seed42 logits differ')
    require([row['step'] for row in schedule] == list(range(1,33)), 'Exactly32 scheduled batches required')
    result = []
    for index, step in enumerate(schedule):
        require([p['region'] for p in step['patches']] == [0,1], 'Region order differs')
        fields = {key:[] for key in ['input','manual_mask','manual_hard_target','pseudo_mask','pseudo_hard_target']}
        exposures = []
        for patch in step['patches']:
            region = patch['region']; data, mask, target, pseudo, pseudo_target = regions[region]
            x,y,w,h = patch['bbox']; code = patch['augmentation']
            require(w == h == 128 and patch['bbox'] in base['exposure'][region]['eligible_tiles']
                    and type(code) is int and 0 <= code < 8, 'Patch/augmentation differs')
            image = np.ascontiguousarray(context['normalize_robust'](data['surface'][2:19,y:y+h,x:x+w]), dtype=np.float32)
            require(np.isfinite(image).all(), 'Nonfinite normalized training input')
            fields['input'].append(mechanics.augment(image, code)[None])
            for key, values in [('manual_mask',mask),('manual_hard_target',target),
                                ('pseudo_mask',pseudo),('pseudo_hard_target',pseudo_target)]:
                fields[key].append(mechanics.augment(values[y:y+h,x:x+w], code)[None])
            exposure = dict(region=region)
            for name,m,t in [('manual',mask,target),('pseudo',pseudo,pseudo_target)]:
                crop_m=m[y:y+h,x:x+w];crop_t=t[y:y+h,x:x+w]
                exposure[name]=dict(positive=int((crop_m & (crop_t==1)).sum()), negative=int((crop_m & (crop_t==0)).sum()))
            require(min(exposure['manual'].values()) >= 256, 'Manual class support changed')
            exposures.append(exposure)
        tensors = {key:torch.from_numpy(np.stack(values)) for key,values in fields.items()}
        identity = state_hash(tensors)
        for log in histories.values():
            old = log[index]
            require(old['step'] == index+1 and old['patches'] == step['patches']
                    and old['exposures'] == exposures and old['input_target_sha256'] == identity,
                    'Exact historical input/target/schedule fingerprint differs')
        representation = features(tensors['input'], arm)
        teacher = torch.from_numpy(np.array(teachers[0][index], copy=True))
        result.append(dict(step=index+1, patches=step['patches'], exposures=exposures,
                           input_target_sha256=identity,
                           input_sha256=hashlib.sha256(tensors['input'].numpy().tobytes()).hexdigest(),
                           feature_sha256=hashlib.sha256(representation.numpy().tobytes()).hexdigest(),
                           manual_sha256=state_hash({k:tensors[k] for k in ['manual_mask','manual_hard_target']}),
                           image=tensors['input'], representation=representation, teacher=teacher,
                           mask=tensors['manual_mask'], target=tensors['manual_hard_target']))
    return result


class CachedTileModel:
    def __init__(self, baseline, baseline_metadata, adapter, arm):
        self.baseline = baseline
        self.baseline_metadata = baseline_metadata
        self.adapter = adapter
        self.arm = arm
        self.logits=[]; self.residuals=[]; self.inputs=[]; self.feature_hashes=[]

    def __call__(self, image):
        from depth_adapter import compose
        i=len(self.logits)
        require(i < 25 and image.shape == (1,1,17,128,128) and image.dtype == torch.float32,
                'Unexpected tile count/input')
        digest=hashlib.sha256(image.numpy().tobytes()).hexdigest()
        require(digest == self.baseline_metadata['captured_tiles'][i]['input_sha256'],
                'Cached backbone normalized tile input differs')
        base=torch.from_numpy(np.array(self.baseline[i],copy=True))[None,None]
        representation=features(image,self.arm)
        logits,residual=compose(self.adapter,representation,base)
        require(hashlib.sha256(image.numpy().tobytes()).hexdigest()==digest,'Adapter mutated input')
        self.logits.append(logits.numpy()[0,0].copy());self.residuals.append(residual.numpy()[0,0].copy())
        self.inputs.append(digest);self.feature_hashes.append(hashlib.sha256(representation.numpy().tobytes()).hexdigest())
        return {'ink':logits}


def development_case(plan, case, context, adapter, arm):
    before = state_hash(adapter.state_dict())
    original=read_json(plan['inference_base_plan'])
    original_case=next(c for c in original['cases'] if c['id']==case['id'])
    require(all(case[key]==value for key,value in original_case.items())
            and original['initializer']==plan['backbone_checkpoint_lineage']
            and original['models']['frozen_seed42']['state_sha256']==plan['backbone_state_sha256'],
            'Development source-plan/initializer/case identity differs')
    baseline_metadata = read_json(case['baseline_metadata'])
    baseline_result=read_json(case['baseline_result'])
    require(baseline_result['status']=='PASS' and baseline_result['role']=='frozen_seed42'
            and baseline_result['state_sha256']==plan['backbone_state_sha256']
            and baseline_result['plan']==plan['inference_base_plan']
            and baseline_result['runs'][case['id']]==baseline_metadata,
            'Baseline source result does not authenticate this tile cache')
    baseline = load_array(baseline_metadata['logits'], (25,128,128), np.float32)
    surface = load_array(case['arrays']['surface'], (17,384,384), np.float32)
    primary = load_array(case['arrays']['primary_mask'], (384,384), np.bool_)
    require(baseline_metadata['adapter_layer_start']==0 and baseline_metadata['adapter_direction']=='forward',
            'Cached depth direction/start differs')
    expected_boxes=[[x,y,128,128] for y in range(0,257,64) for x in range(0,257,64)]
    require([row['bbox_roi_xywh'] for row in baseline_metadata['captured_tiles']] == expected_boxes
            and [row['bbox_roi_xywh'] for row in baseline_metadata['inference']['tiles'] if row['status']=='scored'] == expected_boxes,
            'Cached tile placement/order differs')
    capture=CachedTileModel(baseline,baseline_metadata,adapter,arm)
    scores,metadata,evaluated=context['reference'].infer_surface(capture,surface,0,'forward',context['normalize_robust'],None)
    require(metadata['tile_count']==metadata['scored_tile_count']==len(capture.logits)==25
            and [row['bbox_roi_xywh'] for row in metadata['tiles']] == expected_boxes
            and evaluated[primary].all() and np.isfinite(scores[primary]).all(), 'Development coverage differs')
    require(state_hash(adapter.state_dict())==before and all(p.grad is None for p in adapter.parameters()),
            'Development inference mutated state/gradients')
    return scores,evaluated,metadata,capture
