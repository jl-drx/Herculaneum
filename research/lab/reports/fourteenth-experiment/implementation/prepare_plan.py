"""Source/JSON-only plan for both final states; root-ready only after review."""
import argparse
from pathlib import Path
from calibration_common import HERE,LAB,PREVIOUS,ARMS,ROLES,NEGATIVE_COUNTS,LIMITS,ANALYSIS_LIMITS,THRESHOLD_CONTRACT,DESIGN_SHA,BASELINE_CUTOFF,require,record,identity,read_json,write_json,collect

EIGHTH=dict(path=str(LAB/'reports/eighth-experiment/implementation/final-plan/training-plan.json'),size_bytes=76448,
    sha256='eadb8406f594446e62840366c280358ded25b93874bd04e07b3e0e2482ee96f3')
RETENTION=dict(path=str(LAB/'reports/thirteenth-experiment/retention/final-plan/retention-plan.json'),size_bytes=199788,
    sha256='eb75f743a99d9707219689db1893df40947a7467b98d8c326b527c9f5b51f47a')
METRICS=dict(path=str(LAB/'reports/thirteenth-experiment/retention/analysis/metrics.json'),size_bytes=48318,
    sha256='4f1d8df047dc034cfaed3ddce397191d152674921baca6c7e97acd1c692737c3')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--review',type=Path,action='append',default=[]); parser.add_argument('--ready',action='store_true')
    args=parser.parse_args(); eighth=read_json(EIGHTH); retention=read_json(RETENTION); metrics=read_json(METRICS)
    require(retention['ready_for_execution'] is True and retention['eligible_arms']==ARMS and metrics['status']=='PASS'
        and metrics['plan']==RETENTION and metrics['eligible_arms']==ARMS,'Both retained final states required')
    require(retention['models']['A_preserve']['checkpoint']['sha256']=='f91cef62694c51eab240fe6501cc6f49d1a41db6fa24be3307a5e705900f5bdb'
        and retention['models']['B_preserve']['checkpoint']['sha256']=='3b28cc5c7d511142afc557599e74a65f46e7d1992c94db4dc9f3a3355a35cdcb','Final checkpoint pins differ')
    baseline_lock=retention['baseline_cutoff']['lock']; lock=read_json(baseline_lock)
    baseline_result=retention['historical']['frozen_seed42']['pooled_result']; baseline=read_json(baseline_result)
    require(baseline['status']=='PASS' and baseline['arm']=='frozen_seed42' and baseline['plan']==EIGHTH
        and baseline['initial_state_sha256']==retention['initial_state_sha256'],'Baseline initialization lineage differs')
    require(baseline_lock['sha256']=='be82101811a8456caf75f7cf48fff947d64ae3ad0430345069fd1a6165c995cd'
        and lock['primary']==lock['secondary']==BASELINE_CUTOFF and lock['arm']=='frozen_seed42','Baseline literal lock differs')
    maps={name:{} for name in ROLES}; old_workers={arm:read_json(metrics['model_results'][arm]) for arm in ARMS}
    for case in retention['cases']:
        maps['frozen_seed42'][case['id']]=case['historical_maps']['frozen_seed42']
        for arm in ARMS:
            worker=old_workers[arm]; require(worker['status']=='PASS' and worker['plan']==RETENTION
                and worker['state_sha256']==retention['models'][arm]['state_sha256'],'Diagnostic state provenance differs')
            maps[arm][case['id']]=worker['runs'][case['id']]
        for role in ROLES:
            require(identity(maps[role][case['id']]['scores'])==identity(metrics['per_context'][case['id']]['models'][role]['raw_scores'])
                and identity(maps[role][case['id']]['evaluated'])==identity(metrics['per_context'][case['id']]['models'][role]['evaluated']),
                'Reused diagnostic score/coverage pin differs')
    require(not args.ready or args.review,'Ready plan needs technical review')
    plan=dict(kind='development-only-calibration-control-v1',ready_for_execution=args.ready,arms=ARMS,roles=ROLES,
        eighth_plan=EIGHTH,retention_plan=RETENTION,retention_metrics=METRICS,
        development=eighth['development'],diagnostic_cases=retention['cases'],diagnostic_maps=maps,
        diagnostic_result_records=metrics['model_results'],models=retention['models'],initializer=eighth['initializer'],
        initial_state_sha256=retention['initial_state_sha256'],helper=eighth['helper'],shared=retention['shared'],
        metrics=eighth['metrics'],cutoff=eighth['cutoff'],legacy_prepare=retention['legacy_prepare'],legacy_worker=retention['legacy_worker'],
        retention_common=record(PREVIOUS),baseline_lock=baseline_lock,baseline_result=baseline_result,baseline_cutoff=BASELINE_CUTOFF,
        negative_counts=NEGATIVE_COUNTS,
        research_design=record(HERE.parent/'RESEARCH_DESIGN.md'),threshold_contract=THRESHOLD_CONTRACT,
        inference=dict(layer_start=2,direction='forward',mask_passed_to_adapter=False,patch_shape=[17,128,128],stride=64,
            tiles_by_case=[9,25,25],forwards_per_arm=59,normalization='unchanged robust_mad 1/99',baseline_forwards=0,diagnostic_forwards=0,backwards=0),
        limits=LIMITS,analysis_limits=ANALYSIS_LIMITS,reviews=[record(p) for p in args.review],
        interpretation='Calibration-only descriptive transfer on reused development/retention evidence; two states, no state selection, no new training, independent accuracy, reading, significance, automatic promotion or paid compute.')
    require(plan['research_design']['sha256']==DESIGN_SHA,'Root design changed')
    # Retain the entire focused vendor source manifest pins from the original ready plan.
    plan['transitive_runtime_inputs']=[item for item in eighth['verified_inputs']
        if str(Path(item['path'])).startswith(str(LAB/'vendor')+'/') or Path(item['path']).name=='run_ink_reference.py']
    require(any(Path(item['path']).name=='source-manifest.json' for item in plan['transitive_runtime_inputs']), 'Vendor source manifest pin absent')
    manifest_record=next(item for item in plan['transitive_runtime_inputs'] if Path(item['path']).name=='source-manifest.json')
    manifest=read_json(manifest_record);vendor=Path(manifest_record['path']).parent
    require(manifest['source_commit']=='e0bbb8b40a2db58b1d71864f286eb85717e59e64','Focused upstream revision differs')
    for item in manifest['files']:
        path=(vendor/item['path']).resolve();require(path.is_relative_to(vendor),'Vendor source escapes pinned root')
        plan['transitive_runtime_inputs'].append(dict(path=str(path),size_bytes=item['size_bytes'],sha256=item['sha256']))
    license_record=record(vendor/'UPSTREAM-LICENSE.txt');require(license_record['sha256']==manifest['license_sha256'],'Vendor license changed')
    plan['transitive_runtime_inputs'].append(license_record)
    plan['code']={p.name:record(p) for p in sorted(HERE.glob('*.py'))}
    plan['code_freeze']=record(HERE/'code-freeze.json') if args.ready else None
    pins={}; collect(plan,pins); plan['verified_inputs']=list(pins.values())
    args.output_dir.mkdir(parents=True,exist_ok=False); target=args.output_dir/'calibration-plan.json';write_json(target,plan)
    print(__import__('json').dumps(dict(status='PASS',plan=record(target),ready=args.ready,actual_array_reads=0,actual_checkpoint_reads=0,model_calls=0)))


if __name__=='__main__':main()
