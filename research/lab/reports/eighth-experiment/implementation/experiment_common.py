"""Local frozen inputs and unchanged upstream sources for the eighth pilot."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ['frozen_seed42', 'A', 'B', 'C']


def require(condition, message):
    if not condition: raise ValueError(message)


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def verify(item):
    require(record(item['path']) == item, 'File identity mismatch: ' + Path(item['path']).name)
    return Path(item['path'])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def module(item, name):
    path = verify(item); spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def load_plan(path, execute=False):
    path = Path(path).resolve(strict=True); plan = json.loads(path.read_text())
    require(plan['kind'] == 'matched-continuation-v1', 'Wrong prototype plan')
    if execute: require(plan['ready_for_execution'] is True, 'Prototype not frozen ready')
    for item in plan['verified_inputs']: verify(item)
    for name, item in plan['code'].items(): require(verify(item) == HERE/name, 'Execution source differs')
    require(plan['training'] == dict(steps=32, batch_size=2, seed=20261008, learning_rate=.0001,
        optimizer='AdamW', betas=[.9,.999], eps=1e-8, weight_decay=0, scheduler=None,
        gradient_clip_norm=1., teacher_threshold=.5, loss_smoothing=0,
        manual_coefficient=1., pseudo_coefficient=.25, balanced_manual_mixture=.5), 'Training contract changed')
    require([c['id'] for c in plan['adaptation']] == ['pherc0139-w016-rank01','pherc0139-w016-rank02'], 'Adaptation scope changed')
    require([c['id'] for c in plan['development']]==['pherc1667-w029','pherc0814-46527-rank01','pherc0814-46527-rank07']
        and len(plan['fresh'])==2 and all(c['physical_group']=='PHerc0814' for c in plan['fresh']), 'Development/fresh scope differs')
    require(plan['limits'] == dict(wall_seconds=120, cpu_seconds=360, threads=4,
        rss_acceptance_bytes=4*1024**3, address_space_bytes=12*1024**3,
        output_bytes=700*1024**2, retries=0), 'Resource cap changed')
    require(plan['initializer']['label']=='seed42' and plan['teacher']['label']=='seed43', 'Fixed roles changed')
    require(plan['gate']==dict(comparators=['B','A','frozen_seed42'],specificity_min=.95,mean_sensitivity_gain_min=.02,
        per_context_sensitivity_delta_min=-.02,per_context_auc_delta_min=-.01,per_context_ap_delta_min=-.02,
        promising_requires_all_comparisons=True,statistical_significance_claim=False),'Predeclared gate changed')
    verify(plan['schedule']); verify(plan['helper']); verify(plan['metrics']); verify(plan['cutoff'])
    return plan, path


def arrays(case):
    result = {key: np.load(verify(value), mmap_mode='r', allow_pickle=False) for key,value in case['inputs'].items()}
    surface = result['surface']; require(surface.dtype==np.uint8 and surface.shape[0]==21
        and surface.shape[1:] in [(256,256),(384,384)], 'Pooled source shape differs')
    observed=result['acquisition_observed_mask']; require(observed.dtype==np.bool_ and observed.shape==surface.shape[1:]
        and observed.all(), 'Unobserved source pixels; no fill allowed')
    for key,array in result.items():
        if key!='surface': require(array.shape==surface.shape[1:] and array.dtype in [np.dtype('bool'),np.dtype('uint8')], 'Aligned mask/annotation differs')
    require(result['source_observed_nonzero_mask'].dtype==np.bool_, 'Occupancy dtype differs')
    require(np.array_equal(result['source_observed_nonzero_mask'],np.any(surface[2:19]!=0,axis=0)), 'Occupancy differs')
    return result


def training_region(case):
    data=arrays(case); shape=data['surface'].shape[1:]
    require(case['physical_group']=='PHerc0139' and shape==(384,384), 'No other specimen may adapt')
    supervision=data['supervision_mask']>0; validation=data['validation_mask']>0
    manual=supervision & ~validation & data['acquisition_observed_mask']
    raw=data['inklabels']; require(raw.dtype==np.uint8 and set(np.unique(raw[manual]).tolist())<={0,1}, 'Manual encoding differs')
    targets=np.full(shape,-1.,dtype=np.float32); targets[manual]=raw[manual]  # No retired V ink targets accessed.
    teacher=np.load(verify(case['teacher_scores']),allow_pickle=False)
    evaluated=np.load(verify(case['teacher_evaluated']),allow_pickle=False)
    require(teacher.dtype==np.float32 and teacher.shape==shape and evaluated.dtype==np.bool_ and evaluated.shape==shape, 'Teacher map differs')
    finite=np.isfinite(teacher); require(np.all((teacher[finite]>=0)&(teacher[finite]<=1)), 'Invalid teacher values')
    pseudo=data['acquisition_observed_mask'] & evaluated & finite & ~manual  # Retired V deliberately eligible.
    pseudo_targets=np.full(shape,-1.,dtype=np.float32); pseudo_targets[pseudo]=(teacher[pseudo]>=.5).astype(np.float32)
    return data,manual,targets,pseudo,pseudo_targets


def evaluation(case):
    data=arrays(case); labels=data['inklabels']; role=case['evaluation_participation_role']
    require(role in ['public_validation','supplied_supervision'], 'Evaluation role unspecified')
    participation=data['evaluation_participation_mask']>0; shape=labels.shape
    expected=data['validation_mask']>0 if role=='public_validation' else data['supervision_mask']>0
    require(np.array_equal(participation,expected),'Evaluation participation identity differs')
    require(labels.dtype==np.uint8 and set(np.unique(labels[participation]).tolist())<={0,1}, 'Reference encoding differs')
    primary=participation & data['acquisition_observed_mask']; margin=np.zeros(shape,dtype=bool)
    margin[64:shape[0]-64,64:shape[1]-64]=True; primary &= margin
    positive=int((primary&(labels>0)).sum()); negative=int((primary&(labels==0)).sum())
    require(positive>=256 and negative>=256,'Fixed primary lacks256 of each reference class')
    return data,labels>0,primary


def state_hash(state):
    digest=hashlib.sha256()
    for key in sorted(state):
        tensor=state[key].detach().cpu().contiguous()
        digest.update(json.dumps([key,list(tensor.shape),str(tensor.dtype)],separators=(',',':')).encode()+b'\0')
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
