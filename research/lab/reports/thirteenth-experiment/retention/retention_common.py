"""Narrow immutable contracts for conditional final-state retention."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LAB = Path('/workspace/herculaneum-lab')
ARMS = ['A_preserve', 'B_preserve']
CASE_IDS = ['pherc0814-46527-fresh-rank01', 'pherc0814-46527-fresh-rank14', 'frag4-context1', 'frag4-context2']
LIMITS = dict(threads=4, worker_wall_seconds=120, worker_cpu_seconds=360, total_wall_seconds=300,
    rss_bytes=3*1024**3, address_space_bytes=12*1024**3, output_bytes=64*1024**2,
    maximum_tile_forwards_per_arm=100, maximum_tile_forwards=200, retries=0)
ANALYSIS_LIMITS = dict(wall_seconds=120,cpu_seconds=120,rss_bytes=1024**3,address_space_bytes=4*1024**3)
GATE = dict(minimum_improvement=1e-4, base_and_full_required=True, fixed_exposures=32,
    scope='Same frozen training schedule only; no retention scores or labels select eligibility',
    delta_arithmetic='final equal-batch mean minus initializer equal-batch mean; inclusive <= -1e-4; no gate tolerance')
SCREEN = dict(domain_macro_auc_delta_min=-.01, domain_macro_ap_delta_min=-.01,
    per_context_auc_delta_min=-.02, per_context_ap_delta_min=-.02, automatic_promotion=False)
ROOT_DESIGN_SHA = '59bee53d3f1a662b37aec43fde31698ce0a9946ea0dd0de7027c01c3cf254f36'


def require(condition, message):
    if not condition: raise ValueError(message)


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def identity(item):
    sizes = [item[k] for k in ['size_bytes', 'byte_length', 'bytes'] if k in item]
    require(sizes and all(type(n) is int and n == sizes[0] for n in sizes), 'Invalid size identity')
    require(isinstance(item['sha256'], str) and len(item['sha256']) == 64, 'Invalid SHA identity')
    return dict(path=str(Path(item['path']).resolve()), size_bytes=sizes[0], sha256=item['sha256'])


def verify(item):
    expected = identity(item)
    require(record(expected['path']) == expected, 'File identity changed: '+Path(expected['path']).name)
    return Path(expected['path'])


def read_json(item): return json.loads(verify(item).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def module(item, name):
    spec = importlib.util.spec_from_file_location(name, verify(item))
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result; spec.loader.exec_module(result)
    return result


def collect(value, records):
    """Collect identities from metadata without opening arrays/checkpoints."""
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            item = identity(value)
            require(item['path'] not in records or records[item['path']] == item, 'Conflicting metadata identities')
            records[item['path']] = item
        else:
            for child in value.values(): collect(child, records)
    elif isinstance(value, list):
        for child in value: collect(child, records)


def gate_eligible(measured):
    require(measured['exposures'] == 32, 'Gate exposure count differs')
    for key in ['base_initial','base_final','full_initial','full_final']:
        require(type(measured[key]) in [int,float] and math.isfinite(measured[key]), 'Gate loss must be finite numeric')
    base_delta=measured['base_final']-measured['base_initial']
    full_delta=measured['full_final']-measured['full_initial']
    require(measured['base_delta']==base_delta and measured['full_delta']==full_delta, 'Gate delta arithmetic differs')
    return base_delta <= -1e-4 and full_delta <= -1e-4


def verify_gate_result(entry, arm, training_plan, initial_state_sha256):
    """Bind the gate to the actual saved worker JSON, including failed arms."""
    expected=gate_eligible(entry['fixed_evaluation']); receipt=read_json(entry['training_result'])
    require(receipt['status']=='PASS' and receipt['arm']==arm and identity(receipt['plan'])==identity(training_plan)
        and receipt['initial_state_sha256']==initial_state_sha256, 'Training result lineage differs')
    require(receipt['eligible'] is expected and entry['eligible'] is expected, 'Worker/gate eligibility differs')
    require(identity(receipt['checkpoint'])==identity(entry['checkpoint'])
        and receipt['final_state_sha256']==entry['final_state_sha256'], 'Gate checkpoint differs')
    require(receipt['actual_calls']==dict(student_forwards=192,teacher_forwards=0,backwards=32,optimizer_steps=32), 'Training call scope differs')
    for name,risk in [('initial',receipt['baseline_fixed_evaluation']),('final',receipt['fixed_evaluation']['32'])]:
        require(len(risk['per_batch'])==32, 'Fixed risk must have 32 equal batch exposures')
        for objective in ['base','full']:
            values=[row[objective] for row in risk['per_batch']]
            require(all(type(v) in [int,float] and math.isfinite(v) for v in values), 'Fixed batch risk must be finite numeric')
            mean=math.fsum(values)/32
            require(risk['equal_batch_means'][objective]==mean
                and entry['fixed_evaluation'][objective+'_'+name]==mean, 'Gate mean differs from worker fixed risk')
    require(receipt['fixed_evaluation']['32']['state_sha256']==entry['final_state_sha256'], 'Final fixed-risk state differs')
    return expected


def load_plan(path, execute=True):
    path = Path(path).resolve(strict=True); plan = json.loads(path.read_text())
    require(plan['kind'] == 'conditional-final-state-retention-v1', 'Wrong retention plan')
    if execute: require(plan['ready_for_execution'] is True, 'Conditional retention not authorized ready')
    require(plan['arms'] == ARMS and [c['id'] for c in plan['cases']] == CASE_IDS, 'Fixed retention scope differs')
    require(plan['limits'] == LIMITS and plan['gate_contract'] == GATE and plan['screen_contract']==SCREEN, 'Fixed limits/gate/screen differ')
    require(plan['research_design']['sha256']==ROOT_DESIGN_SHA, 'Scientific design pin differs')
    require(plan['analysis_limits']==ANALYSIS_LIMITS, 'Analysis resource limits differ')
    require(plan['baseline_cutoff']['value']==0.7558947205543518
        and all(c['fixed_baseline_cutoff']==plan['baseline_cutoff']['value'] for c in plan['cases']), 'Common frozen baseline cutoff differs')
    require(plan['inference'] == dict(direction='forward', mask_passed_to_adapter=False,
        normalization='unchanged retained robust_mad 1/99 per 17x128x128 tile', tiles_per_case=25,
        no_baseline_forward=True, no_training=True, no_threshold_fitting=True), 'Inference scope differs')
    require(plan['eligible_arms'] == [a for a in ARMS if a in plan['models']], 'Eligible role order differs')
    if execute:
        require(plan['training_gate'] is not None and plan['root_decision'] is not None and plan['reviews'], 'Missing authorization receipts')
        gate = read_json(plan['training_gate'])
        require(gate['status'] == 'PASS' and gate['gate_contract'] == GATE, 'Training gate contract differs')
        require(identity(gate['plan'])==plan['training_plan'], 'Training plan lineage differs')
        require(gate['eligible_arms'] == plan['eligible_arms'], 'Eligibility changed')
        for arm in ARMS:
            entry = gate['arms'][arm]; measured = entry['fixed_evaluation']
            expected = verify_gate_result(entry,arm,plan['training_plan'],plan['initial_state_sha256'])
            require(entry['eligible'] is expected and expected == (arm in plan['models']), 'Training-only gate is inconsistent')
            if expected:
                model = plan['models'][arm]
                require(model['checkpoint'] == identity(entry['checkpoint'])
                    and model['state_sha256'] == entry['final_state_sha256']
                    and model['training_result'] == identity(entry['training_result']), 'Eligible state binding differs')
    for item in plan['verified_inputs']: verify(item)
    for name, item in plan['code'].items(): require(verify(item) == HERE/name, 'Retention execution source differs')
    if execute:
        freeze = read_json(plan['code_freeze'])
        require(freeze['no_further_edits'] is True, 'Retention source not sealed')
        for item in freeze['files']:
            verify(item)
            if Path(item['path']).suffix == '.py':
                require(plan['code'][Path(item['path']).name] == item, 'Plan differs from code freeze')
    return plan, path


def load_case(plan, case):
    """Reuse each historical input/mask adapter; never resample or shrink support."""
    import numpy as np
    if case['family'] == 'pooled21':
        shared = module(plan['shared'], 'thirteenth_pooled_contract')
        data, labels, primary = shared.evaluation(case['legacy_case'])
        require(case['layer_start'] == 2 and data['surface'].dtype == np.uint8
            and data['surface'].shape == (21,384,384), 'Legacy pooled input differs')
        surface = data['surface']
    else:
        require(case['family'] == 'frag4_float17' and case['layer_start'] == 0, 'Unknown input family')
        legacy = module(plan['legacy_prepare'], 'thirteenth_frag4_contract')
        data = legacy.load_arrays(case['legacy_case'])
        surface, labels, primary = data['surface'], data['labels'].astype(bool), data['primary_mask']
    fixed = np.load(verify(case['primary_mask']), mmap_mode='r', allow_pickle=False)
    require(fixed.dtype == np.bool_ and fixed.shape == (384,384) and np.array_equal(primary,fixed), 'Canonical mask changed')
    require(labels.dtype == np.bool_ and labels.shape == fixed.shape, 'Label shape/encoding differs')
    return surface, labels, fixed
