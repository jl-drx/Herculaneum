"""Freeze five retained states and cutoffs for a supplied Frag4 reference comparison."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = Path('/workspace/herculaneum-lab')
ROLES = ['frozen_seed42', 'A_smooth', 'B_smooth', 'A_preserve', 'B_preserve']
DIRECTIONS = ['forward', 'reverse']
TENTH_PLAN = dict(path=str(LAB/'reports/tenth-experiment/implementation/final-plan/control-plan.json'),
    size_bytes=109068, sha256='9e2c9f4f778c3f794e103191fd413937b521fa0d0b72cab0e3905778ce3e3a55')
TENTH_METRICS = dict(path=str(LAB/'reports/tenth-experiment/implementation/analysis/metrics.json'),
    size_bytes=57942, sha256='ff8e549c93ae1430373ff956a5448b5e10abf7202ede858828b0f8192362b751')
PREPROCESSING = dict(source_planes_inclusive=[7,57], depth_group_size=3, xy_group_size=3,
    sum_dtype='uint32', division_dtype='float64', output_dtype='float32', scale_divisor=257,
    formula='float32(uint32_sum_of_27_source_uint16_values / float64(27*257))',
    source_group_centers=list(range(8,57,3)), center_source_plane=32, pixel_spacing_um=9.72,
    integer_quantization=False)


def require(condition, message):
    if not condition: raise ValueError(message)


def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)


def identity(item):
    sizes=[item[key] for key in ['size_bytes','byte_length','bytes'] if key in item]
    require(sizes and all(type(n) is int and n==sizes[0] for n in sizes),'Invalid file size record')
    return dict(path=str(Path(item['path']).resolve()),size_bytes=sizes[0],sha256=item['sha256'])


def verify(item):
    expected=identity(item); require(record(expected['path'])==expected,'File identity changed: '+Path(expected['path']).name)
    return Path(expected['path'])


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def module(item, name):
    spec=importlib.util.spec_from_file_location(name,verify(item)); result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result; spec.loader.exec_module(result); return result


def collect(value, records):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            item=identity(value); verify(item)
            require(item['path'] not in records or records[item['path']]==item,'Conflicting file pins')
            records[item['path']]=item
        else:
            for child in value.values(): collect(child,records)
    elif isinstance(value,list):
        for child in value: collect(child,records)


def load_arrays(case):
    arrays={key:np.load(verify(item),mmap_mode='r',allow_pickle=False) for key,item in case['arrays'].items()}
    require(set(arrays)=={'surface','labels','primary_mask','known_mask','surface_mask'},'Reference array roles differ')
    surface=arrays['surface']; require(surface.dtype==np.float32 and surface.shape==(17,384,384)
        and np.isfinite(surface).all() and np.all((surface>=0)&(surface<=255)),'Pooled float32 input contract differs')
    labels=arrays['labels']; require(labels.shape==(384,384) and labels.dtype in [np.dtype('uint8'),np.dtype('bool')]
        and set(np.unique(labels).tolist())<={0,1},'Supplied label encoding differs')
    for name in ['primary_mask','known_mask','surface_mask']:
        require(arrays[name].dtype==np.bool_ and arrays[name].shape==(384,384),'Reference mask dtype/shape differs')
    known=arrays['known_mask']; require(np.all(~known|arrays['surface_mask']),'Known reference escapes surface mask')
    border=np.zeros((384,384),dtype=bool); border[32:-32,32:-32]=True
    require(np.array_equal(arrays['primary_mask'],known&border),'Fixed32px primary mask differs')
    primary=arrays['primary_mask']; require(int((primary&(labels==1)).sum())>=512
        and int((primary&(labels==0)).sum())>=8192,'Fixed primary class gate failed; no relaxation')
    return arrays


def load_plan(path, execute=True):
    path=Path(path).resolve(strict=True); plan=json.loads(path.read_text())
    require(plan['kind']=='Frag4-five-state-reference-comparison-v1','Wrong comparison kind')
    if execute: require(plan['ready_for_execution'] is True,'Comparison not ready')
    require(plan['roles']==ROLES and plan['directions']==DIRECTIONS and plan['preprocessing']==PREPROCESSING,'Frozen scope differs')
    require([c['id'] for c in plan['cases']]==['frag4-context1','frag4-context2'],'Fixed contexts differ')
    require(plan['inference']==dict(layer_start=0,layer_count=17,patch_shape=[17,128,128],stride=64,
        normalization='unchanged robust_mad 1/99 from retained factory',mask_passed_to_adapter=False,
        forward_primary=True,reverse_secondary=True,best_orientation_selection=False),'Inference contract differs')
    require(plan['limits']==dict(threads=4,total_wall_seconds=900,worker_wall_seconds=120,worker_cpu_seconds=360,
        rss_bytes=3*1024**3,address_space_bytes=12*1024**3,maximum_tile_forwards=500,
        output_bytes=200*1024**2,retries=0),'Resource limits differ')
    for item in plan['verified_inputs']: verify(item)
    for name,item in plan['code'].items(): require(verify(item)==HERE/name,'Comparison code path changed')
    freeze=json.loads(verify(plan['code_freeze']).read_text())
    for item in freeze['files']:
        verify(item)
        if Path(item['path']).suffix=='.py':
            require(plan['code'][Path(item['path']).name]==item,'Code differs from sealed package')
    for role in ROLES:
        require(plan['models'][role]['role']==role and set(plan['models'][role]['thresholds'])=={'primary','secondary'},'State/threshold mapping differs')
    return plan,path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--review',type=Path,action='append',default=[]); parser.add_argument('--ready',action='store_true')
    args=parser.parse_args(); base=json.loads(verify(TENTH_PLAN).read_text()); metrics=json.loads(verify(TENTH_METRICS).read_text())
    require(metrics['status']=='PASS' and metrics['plan']==TENTH_PLAN,'Retained comparison receipt differs')
    descriptor_record=record(args.descriptor); descriptor=json.loads(verify(descriptor_record).read_text())
    require(descriptor['status']=='pooled_reference_complete_no_inference','Pooled reference incomplete')
    require(all(descriptor['preprocessing'].get(k)==v for k,v in PREPROCESSING.items()),'Declared pooling/scaling differs')
    require([c['id'] for c in descriptor['cases']]==['frag4-context1','frag4-context2'],'Fixed input IDs differ')
    cases=[]
    for value in descriptor['cases']:
        case=dict(id=value['id'],native_bbox_yxhw=value['native_bbox_yxhw'],arrays={k:identity(v) for k,v in value['arrays'].items()})
        load_arrays(case); cases.append(case)
    models={}
    for role in ROLES:
        result_record=metrics['model_results'][role]; result=json.loads(verify(result_record).read_text())
        require(result['status']=='PASS' and result['arm']==role and result['initial_state_sha256']==base['historical_initial_state_sha256'],'Retained state receipt differs')
        threshold=result['threshold_lock']; require(threshold==metrics['thresholds'][role],'Retained cutoff pin differs')
        lock=json.loads(verify(threshold).read_text()); require(lock['arm']==role and lock['plan']==result['plan'],'Own cutoff provenance differs')
        values={key:lock[key] for key in ['primary','secondary']}
        require(all(type(v) in [int,float] and np.isfinite(v) for v in values.values()),'Nonfinite frozen threshold')
        training=result['training']; checkpoint=None if role=='frozen_seed42' else training['checkpoint']
        models[role]=dict(role=role,result=result_record,checkpoint=checkpoint,threshold_lock=threshold,thresholds=values,
            state_sha256=base['historical_initial_state_sha256'] if checkpoint is None else training['final_state_sha256'])
    shared=record(LAB/'reports/eighth-experiment/implementation/experiment_common.py')
    require(shared['sha256']=='8d933dae02248d7b5fbbd26370c95fca53191bcff4e1dc51e984c40d8095b8cf','Shared state hasher changed')
    require(not args.ready or args.review,'Ready execution requires root-reviewed receipts')
    plan=dict(kind='Frag4-five-state-reference-comparison-v1',ready_for_execution=args.ready,roles=ROLES,directions=DIRECTIONS,
        primary_direction='forward',secondary_direction='reverse',descriptor=descriptor_record,preprocessing=PREPROCESSING,
        cases=cases,models=models,source_plan=TENTH_PLAN,source_results=TENTH_METRICS,initializer=base['initializer'],
        helper=base['helper'],shared=shared,metrics=base['metrics'],cutoff=base['cutoff'],reviews=[record(p) for p in args.review],
        research_design=record(HERE.parent/'RESEARCH_DESIGN.md'),
        inference=dict(layer_start=0,layer_count=17,patch_shape=[17,128,128],stride=64,
            normalization='unchanged robust_mad 1/99 from retained factory',mask_passed_to_adapter=False,
            forward_primary=True,reverse_secondary=True,best_orientation_selection=False),
        limits=dict(threads=4,total_wall_seconds=900,worker_wall_seconds=120,worker_cpu_seconds=360,rss_bytes=3*1024**3,
            address_space_bytes=12*1024**3,maximum_tile_forwards=500,output_bytes=200*1024**2,retries=0),
        interpretation='Conditional supplied-reference generalization diagnostic on2contexts of1fragment; no training/calibration, historical holdout, optical warp authentication, significance, pixel-IID confidence interval or reading claim.',
        complete_output_rule='Preserve all5states x2contexts x2directions and report all; stop on any failure without retry, replacement or model/orientation selection.')
    require(plan['research_design']['sha256']=='5c20b5ffc3e00f6b5342c6ec38090e7236109e9535b493028c3fe1ac205a8d4c','Root scientific design changed')
    plan['code']={p.name:record(p) for p in sorted(HERE.glob('*.py'))}
    plan['code_freeze']=record(HERE/'code-freeze.json')
    pins={}; collect(plan,pins); collect(descriptor,pins); plan['verified_inputs']=list(pins.values())
    args.output_dir.mkdir(parents=True,exist_ok=False); path=args.output_dir/'comparison-plan.json'; write_json(path,plan)
    load_plan(path,execute=args.ready); print(json.dumps(dict(status='PASS',plan=record(path),model_calls=0)))


if __name__=='__main__': main()
