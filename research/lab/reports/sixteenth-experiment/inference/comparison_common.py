"""Narrow offline contracts for fixed-state optical-reference comparison."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROLES = ['frozen_seed42', 'A_preserve', 'B_preserve']
CALIBRATION_IDS = ['frag3-context1', 'frag3-context2']
EVALUATION_IDS = ['frag1-context1', 'frag1-context2', 'frag2-context1', 'frag2-context2']
SOURCE_SHA = '280445f79ee921263657bc7ffb85a4f558df9e0a351ce607dee535e207bf067d'
DESIGN_SHA = '34fd5a0312240a1e27fc5fafa2727c1205e9b2283148fbe27d9a4d70b63f6778'
RUNTIME_SHA = '530aba06cf9f2fde2c16a18731457631416f1f3cdd5c7ec66181ec1c0efef7db'
PREPROCESSING = dict(source_planes_inclusive=[7,57], depth_group_size=3, xy_group_size=3,
    sum_dtype='uint32', division_dtype='float64', output_dtype='float32', scale_divisor=257,
    formula='float32(uint32_sum_of_27_source_uint16_values / float64(27*257))',
    source_group_centers=list(range(8,57,3)), center_source_plane=32, pixel_spacing_um=9.72,
    integer_quantization=False)
LIMITS = dict(threads=4, worker_wall_seconds=120, worker_cpu_seconds=360,
    total_wall_seconds=600, rss_bytes=3*1024**3, address_space_bytes=12*1024**3,
    maximum_tile_forwards=450, output_bytes=96*1024**2, retries=0)
INFERENCE = dict(layer_start=0, direction='forward', patch_shape=[17,128,128], stride=64,
    normalization='unchanged robust_mad1/99 over full17x128x128 patch',
    mask_passed_to_adapter=False, tiles_per_context=25)


def require(condition, message):
    if not condition: raise ValueError(message)


def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)


def identity(item):
    sizes=[item[k] for k in ['size_bytes','byte_length','bytes'] if k in item]
    require(sizes and all(type(n) is int and n==sizes[0] for n in sizes),'File size record differs')
    return dict(path=str(Path(item['path']).resolve()),size_bytes=sizes[0],sha256=item['sha256'])


def verify(item):
    expected=identity(item);require(record(expected['path'])==expected,'File pin changed: '+expected['path'])
    return Path(expected['path'])


def read_json(item): return json.loads(verify(item).read_text())
def write_json(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def module(item,name):
    spec=importlib.util.spec_from_file_location(name,verify(item)); value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def sources():
    item=record(HERE/'SOURCE_PINS.json');require(item['sha256']==SOURCE_SHA,'Retained source pins changed')
    return json.loads(verify(item).read_text()),item


def collect(value,records):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            item=identity(value);verify(item)
            require(item['path'] not in records or records[item['path']]==item,'Conflicting pins')
            records[item['path']]=item
        else:
            for child in value.values():collect(child,records)
    elif isinstance(value,list):
        for child in value:collect(child,records)


def load_plan(path,execute=True):
    path=Path(path).resolve(strict=True);plan=json.loads(path.read_text())
    require(plan['kind']=='Paris-fragments-fixed-state-calibration-transfer-v1','Wrong plan kind')
    require(plan['roles']==ROLES and plan['preprocessing']==PREPROCESSING and plan['limits']==LIMITS
        and plan['inference']==INFERENCE,'Fixed comparison scope differs')
    require([c['id'] for c in plan['cases']]==CALIBRATION_IDS+EVALUATION_IDS,'Case order differs')
    source,source_record=sources();require(plan['source_pins']==source_record,'Source binding differs')
    require(plan['models']==source['models'],'State/cutoff identities differ')
    for name in ['initializer','helper','shared','legacy_prepare','legacy_worker','cutoff','metrics',
            'transitive_runtime_inputs','load_accounting_clarification']:
        require(plan[name]==source[name],'Retained execution source differs: '+name)
    if execute:
        require(plan['ready_for_execution'] is True and plan['descriptor'] is not None,'Plan not ready')
        require(plan['research_design'] is not None and plan['reviews'] and plan['code_freeze'] is not None,'Missing frozen reviews/design')
        require(plan['research_design']['sha256']==DESIGN_SHA,'Root scientific design differs')
        validate_ready_receipts(plan['descriptor'],plan['code_freeze'],plan['research_design'],plan['source_go'],plan['ct_verification'])
        require(plan['runtime']['sha256']==RUNTIME_SHA,'Runtime pin differs')
        environment=read_json(plan['runtime'])
        import importlib.metadata
        require(list(sys.version_info[:3])==environment['python'],'Python version differs')
        for name,version in environment['packages'].items():
            require(importlib.metadata.version(name)==version,'Package version differs: '+name)
        freeze=read_json(plan['code_freeze'])
        require(freeze['no_further_edits'] is True,'Code package not sealed')
        for item in freeze['files']:
            verify(item)
            if Path(item['path']).suffix=='.py':
                require(plan['code'][Path(item['path']).name]==identity(item),'Source differs from sealed code')
        require(set(plan['code'])=={p.name for p in HERE.glob('*.py')},'Executable source set changed')
        for name,item in plan['code'].items():require(verify(item)==HERE/name,'Code path differs')
    for item in plan.get('verified_inputs',[]):verify(item)
    return plan,path


def load_arrays(plan,case):
    original=module(plan['legacy_prepare'],'sixteenth_pinned_array_contract')
    arrays=original.load_arrays(case)
    primary=arrays['primary_mask'];labels=arrays['labels'].astype(bool)
    require(int((primary&labels).sum())==case['primary_positive']
        and int((primary&~labels).sum())==case['primary_negative'],'Fifteenth class support changed')
    return arrays


def require_primary_coverage(scores,evaluated,primary):
    require(scores.dtype==np.float32 and scores.shape==(384,384) and evaluated.dtype==np.bool_
        and evaluated.shape==scores.shape and primary.dtype==np.bool_ and primary.shape==scores.shape,
        'Prediction/support shape or dtype differs')
    require(evaluated[primary].all() and np.isfinite(scores[primary]).all()
        and np.all((scores[primary]>=0)&(scores[primary]<=1)), 'Fixed primary unscored/nonfinite; no shrinking')


def validate_ready_receipts(descriptor,freeze,design,source_go,ct_verification):
    require(all(x is not None for x in [descriptor,freeze,design,source_go,ct_verification]),'Readiness evidence missing')
    go=read_json(source_go);ct=read_json(ct_verification)
    require(go['status']=='GO' and identity(go['code_freeze'])==identity(freeze)
        and identity(go['research_design'])==identity(design),'Source GO does not bind sealed code/design')
    require(ct['status']=='PASS' and ct['all_CT_arrays_exact'] is True
        and identity(ct['descriptor'])==identity(descriptor),'Independent CT PASS does not bind exact descriptor')


def verify_locks_ready(plan,path,expected_sha):
    pin=record(path);require(pin['sha256']==expected_sha,'Calibration barrier changed')
    ready=read_json(pin)
    require(ready['status']=='PASS' and ready['plan']==plan['_plan_record']
        and ready['roles']==ROLES and set(ready['threshold_locks'])==set(ROLES) and ready['calibration_tile_forwards']==150
        and ready['evaluation_tile_forwards']==0,'Incomplete calibration barrier')
    for role in ROLES:
        completed=read_json(ready['models'][role])
        require(completed['status']=='PASS' and completed['role']==role and completed['phase']=='calibration'
            and completed['plan']==plan['_plan_record'] and completed['tile_forward_count']==50
            and completed['state_sha256']==plan['models'][role]['state_sha256']
            and completed['threshold_lock']==ready['threshold_locks'][role],'Calibration completion differs')
        lock=read_json(ready['threshold_locks'][role])
        require(lock['role']==role and lock['state_sha256']==plan['models'][role]['state_sha256']
            and lock['plan']==plan['_plan_record'] and set(lock['regional'])==set(CALIBRATION_IDS), 'Cutoff lineage differs')
        require(np.isfinite(lock['primary']) and all(np.isfinite(x['threshold']) for x in lock['regional'].values()),'Nonfinite calibration cutoff')
        require(lock['primary']==max(x['threshold'] for x in lock['regional'].values()),'Max regional cutoff differs')
        for case in plan['cases'][:2]:
            result=lock['regional'][case['id']]
            require(result['negative_count']==case['primary_negative']
                and result['development_false_positives']<=case['primary_negative']//20,'Calibration support differs')
    return ready,pin


def equal_context_mean(values):
    require(values,'Nonempty macro inputs required')
    if any(x is None for x in values):return None
    require(all(np.isfinite(x) for x in values),'Finite macro inputs required')
    import math
    return math.fsum(float(x) for x in values)/len(values)
