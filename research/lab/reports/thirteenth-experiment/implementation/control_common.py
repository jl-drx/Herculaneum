"""Additive conservative continuation, with immutable tenth mechanics."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEN = HERE.parents[1] / 'tenth-experiment/implementation'
LAB = HERE.parents[2]
OLD_HASH = '9754e1b3445e69214e07b6e865ccad5fa888bd670969c586f0d7f3e9b9bfdd68'
if hashlib.sha256((TEN/'control_common.py').read_bytes()).hexdigest() != OLD_HASH:
    raise ValueError('Immutable tenth contract helper changed')
spec = importlib.util.spec_from_file_location('conservative_tenth_shared', TEN/'control_common.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
record, verify, write_json, module, require, training_region, state_hash = (
    getattr(old, name) for name in ['record','verify','write_json','module','require','training_region','state_hash'])
ARMS = ['A_preserve', 'B_preserve']
UPDATES = [1, 8, 16, 32]
LIMITS = dict(worker_wall_seconds=180, total_wall_seconds=420, cpu_seconds=600, threads=4,
    inter_op_threads=1, rss_acceptance_bytes=4*1024**3, address_space_bytes=12*1024**3,
    output_bytes=450*1024**2, retries=0)
MEASUREMENT = dict(updates=UPDATES, exposures=32,
    reduction='math.fsum of32 logged per-batch scalars /32; BCE FP32; full FP32 base + FP64 KL; deltas final minus initializer',
    baseline='cache-derived initializer risk', training_mode='unchanged', redundant_current_batch=True,
    teacher_forwards=0, forwards_per_arm=192, backwards_per_arm=32, updates_per_arm=32)
GATE = dict(minimum_improvement=1e-4, base_and_full_required=True, fixed_exposures=32,
    delta_arithmetic='final equal-batch mean minus initializer equal-batch mean; inclusive <= -1e-4; no gate tolerance',
    scope='Same frozen training schedule only; no retention scores or labels select eligibility')


def load_plan(path, execute=False):
    path = Path(path).resolve(strict=True); plan = json.loads(path.read_text())
    require(plan['kind']=='conservative-fixed-risk-continuation-v1', 'Wrong conservative control')
    if execute: require(plan['ready_for_execution'] is True, 'Control not ready')
    for item in plan['verified_inputs']: verify(item)
    for name, item in plan['code'].items(): require(verify(item)==HERE/name, 'Execution source differs')
    require(plan['scientific_design']['sha256']=='59bee53d3f1a662b37aec43fde31698ce0a9946ea0dd0de7027c01c3cf254f36', 'Sealed scientific design changed')
    if execute:
        freeze=json.loads(verify(plan['code_freeze']).read_text())
        require(freeze['no_further_edits'] is True and freeze['code']==plan['code']
                and freeze['scientific_design']==plan['scientific_design'], 'Code/design do not match explicit source freeze')
    base, _ = old.load_plan(verify(plan['base_plan']), execute=True)
    for key in ['adaptation','initializer','teacher','schedule','exposure','helper','preservation','historical_initial_state_sha256']:
        require(plan[key]==base[key], 'Matched tenth contract changed: '+key)
    require(plan['training']==dict(base['training'],learning_rate=1e-6), 'Only learning rate may change')
    require(plan['limits']==LIMITS and plan['measurement']==MEASUREMENT and plan['gate']==GATE,
            'Measurement, gate or limits changed')
    require(plan['new_arms']==ARMS and plan['schedule']['sha256']=='87bb53a0a80ba2268b1b2b020342058c2facd44f343c53b8a5ac478da06e40e1', 'Arm/schedule changed')
    require(plan['runtime']==dict(python='3.12.14',numpy='2.5.3',torch='2.12.0+cpu'), 'Runtime contract changed')
    for arm in ARMS:
        historical = plan['historical'][arm]
        result = json.loads(verify(historical['result']).read_text())
        require(result['status']=='PASS' and result['arm']==arm and result['plan']==plan['base_plan']
                and result['initial_state_sha256']==plan['historical_initial_state_sha256'], 'Historical result changed')
        require(historical['teacher']==result['training']['preupdate_logits']['teacher']
                and historical['steps']==result['training']['steps_log'], 'Teacher/log lineage changed')
    return plan, path


def average(rows):
    import math
    require(len(rows)==32, 'Exactly32 fixed batch exposures required')
    require(all(set(row)==set(rows[0]) and all(math.isfinite(v) for v in row.values()) for row in rows),'Nonfinite/inconsistent risk scalars')
    return {key: math.fsum(row[key] for row in rows)/32 for key in rows[0]}


def eligible(initial, final):
    import math
    require(all(math.isfinite(row[key]) for row in [initial,final] for key in ['base','full']), 'Nonfinite gate scalars')
    return final['base']-initial['base'] <= -1e-4 and final['full']-initial['full'] <= -1e-4
