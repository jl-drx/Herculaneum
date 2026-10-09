"""Single offline retained-output calibration verification, without author imports."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys

for option in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[option] = '1'

import numpy as np
from numeric_kernels import (require,select_negatives,lock_values,float32_bits,ranking,operating,
    exact_sensitivity,published_fraction,published_screen,fixtures as numeric_fixtures)
from mask_contracts import (DEVELOPMENT_IDS,NEGATIVE_COUNTS,DIAGNOSTIC_IDS,DIAGNOSTIC_COUNTS,
    development,diagnostic,fixtures as mask_fixtures)
from review_io import Review,record,identity
from blending.blend_verifier import verify_map,verify_normalized_inputs

HERE = Path(__file__).resolve().parent
ARMS = ['A_preserve','B_preserve']
ROLES = ['frozen_seed42']+ARMS
RULES = ['primary','secondary','common_baseline']
BASELINE_CUTOFF = 0.7558947205543518
DESIGN_SHA = '96ed37ff57dbcff80bb9cbd6e984a2bb27ee0f6bf2d7de079dac42adf0b0dc45'
SELECTOR_SHA = '38a0ebbab0835845ee2213c2be9505cf9b41a129ab13dd6668ff797bfacacb73'
BASELINE_LOCK_SHA = 'be82101811a8456caf75f7cf48fff947d64ae3ad0430345069fd1a6165c995cd'
BLEND_SOURCE_SHA = '7a2229ceb595ea4be91dd5f65edefddee7a372120f97677619eb120e1a228858'


def delta(first, second):
    return dict(confusion={k:first['confusion'][k]-second['confusion'][k] for k in ('tp','fp','tn','fn')},
                **{k:first[k]-second[k] if first[k] is not None and second[k] is not None else None
                   for k in ('sensitivity','specificity','precision')})


def macro(rows, role, rule):
    values = [row['models'][role]['operating'][rule] for row in rows]
    return {key:math.fsum(v[key] for v in values)/len(values) if all(v[key] is not None for v in values) else None
            for key in ('sensitivity','specificity','precision')}


def run_review(args, source_pins):
    review = Review()
    entries = {name:record(getattr(args,name)) for name in ('plan','locks_ready','metrics')}
    for name,pin in entries.items():
        require(pin['sha256']==getattr(args,name+'_sha256'),'Root entry pin differs: '+name)
    plan = review.json(entries['plan'])
    require(plan['kind']=='development-only-calibration-control-v1' and plan['ready_for_execution'] is True and
            plan['arms']==ARMS and plan['roles']==ROLES,'Ready fixed calibration plan required')
    require(plan['research_design']['sha256']==DESIGN_SHA and plan['cutoff']['sha256']==SELECTOR_SHA and
            plan['baseline_lock']['sha256']==BASELINE_LOCK_SHA and plan['baseline_cutoff']==BASELINE_CUTOFF,
            'Design/selector/baseline lock pin changed')
    eighth = review.json(plan['eighth_plan'])
    retention = review.json(plan['retention_plan'])
    old = review.json(plan['retention_metrics'])
    require(plan['development']==eighth['development'] and [c['id'] for c in plan['development']]==DEVELOPMENT_IDS and
            plan['negative_counts']==NEGATIVE_COUNTS,'Original literal development cases/counts changed')
    require(plan['diagnostic_cases']==retention['cases'] and [c['id'] for c in plan['diagnostic_cases']]==DIAGNOSTIC_IDS and
            plan['models']==retention['models'] and retention['eligible_arms']==ARMS,'Diagnostic/state selection changed')
    require(old['status']=='PASS' and old['plan']==plan['retention_plan'],'Thirteenth ranking lineage changed')
    require(plan['inference']==dict(layer_start=2,direction='forward',mask_passed_to_adapter=False,
        patch_shape=[17,128,128],stride=64,tiles_by_case=[9,25,25],forwards_per_arm=59,
        normalization='unchanged robust_mad 1/99',baseline_forwards=0,diagnostic_forwards=0,backwards=0),
        'Inference contract changed')
    require(plan['limits']['maximum_tile_forwards']==118 and plan['limits']['retries']==0 and
            plan['threshold_contract']['diagnostic_fitting'] is False,'Budget or diagnostic-fitting scope changed')
    index = review.json(entries['locks_ready'])
    require(index['status']=='PASS' and index['plan']==entries['plan'] and set(index['models'])==set(ARMS) and
            set(index['threshold_locks'])==set(ARMS) and index['tile_forward_count']==118 and index['retries']==0 and
            index['workers']==[dict(arm=a,returncode=0) for a in ARMS] and
            index['both_new_locks_written_before_diagnostic_analysis'] is True and index['diagnostic_arrays_loaded'] is False,
            'Completed once-only locks-ready index required')
    # Both lock artifacts are bound before any diagnostic score or label array decode.
    locks = {arm:review.json(pin) for arm,pin in index['threshold_locks'].items()}
    baseline_lock = review.json(plan['baseline_lock'])
    reconstruction = review.json(index['baseline_reconstruction'])
    workers = {arm:review.json(pin) for arm,pin in index['models'].items()}
    require(baseline_lock['arm']=='frozen_seed42' and baseline_lock['plan']==plan['eighth_plan'] and
            baseline_lock['primary']==baseline_lock['secondary']==BASELINE_CUTOFF,'Frozen baseline literal changed')
    require(reconstruction['status']=='PASS' and reconstruction['original_lock']==plan['baseline_lock'] and
            reconstruction['original_bytes_unchanged'] is True and reconstruction['baseline_forwards']==0 and
            reconstruction['diagnostic_arrays_loaded'] is False,'Baseline reconstruction receipt failed')
    baseline_result = review.json(plan['baseline_result'])
    require(baseline_result['status']=='PASS' and baseline_result['arm']=='frozen_seed42' and
            baseline_result['plan']==plan['eighth_plan'] and plan['initializer']==eighth['initializer']==retention['initializer'] and
            plan['initial_state_sha256']==baseline_result['initial_state_sha256']==retention['initial_state_sha256'],
            'Baseline/model initializer lineage changed')
    expected_states = dict(A_preserve='f91cef62694c51eab240fe6501cc6f49d1a41db6fa24be3307a5e705900f5bdb',
                           B_preserve='3b28cc5c7d511142afc557599e74a65f46e7d1992c94db4dc9f3a3355a35cdcb')
    for arm in ARMS:
        worker,lock = workers[arm],locks[arm]
        require(plan['models'][arm]['checkpoint']['sha256']==expected_states[arm] and worker['status']=='PASS' and
                worker['arm']==lock['arm']==arm and worker['plan']==lock['plan']==entries['plan'] and
                worker['state_sha256']==lock['state_sha256']==plan['models'][arm]['state_sha256'] and
                worker['source_model']==plan['models'][arm] and worker['threshold_lock']==index['threshold_locks'][arm],
                'Final model/lock/worker lineage changed')
        require(worker['tile_forward_count']==59 and worker['actual_calls']==dict(calibration_tile_forwards=59,
                baseline_forwards=0,diagnostic_forwards=0,backwards=0,updates=0) and worker['network_requests']==0 and
                worker['model_state_unchanged'] is True and worker['parameter_gradients_absent'] is True,
                'Worker inference/state/accounting receipt failed')
        require(worker['peak_rss_bytes'] <= plan['limits']['rss_bytes'] and worker['wall_seconds'] <= plan['limits']['worker_wall_seconds'],
                'Worker resource receipt exceeds limit')
        require(set(worker['development'])==set(lock['development'])==set(DEVELOPMENT_IDS) and
                lock['selector']==plan['cutoff'] and lock['diagnostic_arrays_loaded'] is False,'Development-only lock scope changed')
    dev_sources = {case['id']:development(review,case) for case in plan['development']}
    calibration,blend_checks,input_checks = {},{},{}
    for role in ROLES:
        regional = {}
        blend_checks[role],input_checks[role] = {},{}
        for case in plan['development']:
            name = case['id']; labels,mask,surface = dev_sources[name];shape = labels.shape
            item = case['baseline_maps'] if role=='frozen_seed42' else workers[role]['development'][name]
            score = review.array(item['scores'],shape=shape,dtype=np.float32)
            evaluated = review.array(item['evaluated'],shape=shape,dtype=np.bool_)
            require(evaluated[mask].all() and np.isfinite(score[mask]).all() and
                    np.all((score[mask]>=0)&(score[mask]<=1)),'Canonical calibration support unscored/invalid; no shrinking')
            selection = select_negatives(score[mask&~labels])
            actual = baseline_lock['development'][name]['selection'] if role=='frozen_seed42' else locks[role]['development'][name]['selection']
            review.tree(actual,selection,'selection/'+role+'/'+name)
            require(float32_bits(actual['threshold'])==float32_bits(selection['threshold']), 'Cutoff bit pattern differs')
            regional[name] = selection
            if role=='frozen_seed42':
                review.same(item['scores'],baseline_lock['development'][name]['maps']['scores'],'Baseline calibration score pin')
                review.same(item['evaluated'],baseline_lock['development'][name]['maps']['evaluated'],'Baseline calibration coverage pin')
                review.tree(reconstruction['regional'][name],selection,'baseline-reconstruction/'+name)
            else:
                require(locks[role]['development'][name]['maps']==item,'Regional lock/map identity changed')
                saved = review.array(item['fixed_primary_mask'],shape=shape,dtype=np.bool_)
                require(np.array_equal(saved,mask),'Worker calibration mask changed')
                metadata = review.json(record(Path(item['scores']['path']).parent/'metadata.json'))
                require(metadata==item,'Per-case metadata bytes disagree with worker record')
                logits = review.array(item['logits'],shape=(9 if shape==(256,256) else 25,128,128),dtype=np.float32,finite=True)
                blend_checks[role][name] = verify_map(logits,score,evaluated,shape,item)
                input_checks[role][name] = verify_normalized_inputs(surface,shape,item)
                require(item['normalization']==plan['inference']['normalization'] and
                        item['model_input_tensor_unchanged'] is True,'Normalization/input guard receipt changed')
                baseline_meta = review.json(case['baseline_maps']['metadata'])
                old_grid = [t['bbox_roi_xywh'] for t in baseline_meta['inference']['tiles'] if t['status']=='scored']
                require([t['bbox_roi_xywh'] for t in item['captured_tiles']]==old_grid,'Original calibration placements changed')
        expected_locks = lock_values(regional)
        artifact = baseline_lock if role=='frozen_seed42' else locks[role]
        for key,value in expected_locks.items():
            require(artifact[key]==value and float32_bits(artifact[key])==float32_bits(value),'Primary/secondary cutoff bits differ')
        if role=='frozen_seed42':
            require(expected_locks==dict(primary=BASELINE_CUTOFF,secondary=BASELINE_CUTOFF) and
                    reconstruction['primary']==reconstruction['secondary']==BASELINE_CUTOFF,'Baseline reconstruction changed literal lock')
        calibration[role] = dict(regional=regional,**expected_locks,
            primary_bits=float32_bits(expected_locks['primary']),secondary_bits=float32_bits(expected_locks['secondary']))
    for name in DEVELOPMENT_IDS:
        require(input_checks['A_preserve'][name]['records']==input_checks['B_preserve'][name]['records'],
                'Two states saw different normalized calibration inputs')
    report = review.json(entries['metrics'])
    require(report['status']=='PASS' and report['plan']==entries['plan'] and report['locks_ready']==entries['locks_ready'] and
            report['threshold_locks']==index['threshold_locks'] and report['baseline_lock']==plan['baseline_lock'] and
            report['baseline_reconstruction']==index['baseline_reconstruction'] and report['retention_metrics']==plan['retention_metrics'] and
            report['roles']==ROLES and report['model_calls']==report['diagnostic_fitting_calls']==0 and
            report['automatic_promotion'] is False,'Completed diagnostic analysis lineage/scope changed')
    thresholds = {role:dict(primary=calibration[role]['primary'],secondary=calibration[role]['secondary'],
                           common_baseline=BASELINE_CUTOFF) for role in ROLES}
    review.tree(report['thresholds'],thresholds,'reported-thresholds')
    results = {}; ranking_errors = []
    for case in plan['diagnostic_cases']:
        name = case['id']; labels,mask = diagnostic(review,case)
        original = old['per_context'][name]; output = report['per_context'][name]
        require((output['pixel_count'],output['positive'],output['negative'])==DIAGNOSTIC_COUNTS[name] and
                output['physical_group']==case['physical_group'] and set(output['models'])==set(ROLES),
                'Diagnostic canonical counts/group/roles changed')
        review.same(output['fixed_mask'],original['fixed_mask'],'Unchanged diagnostic mask pin')
        saved = review.array(output['fixed_mask'],shape=(384,384),dtype=np.bool_)
        require(np.array_equal(saved,mask),'Saved diagnostic analysis mask changed')
        models = {}
        for role in ROLES:
            item = plan['diagnostic_maps'][role][name]; previous = original['models'][role]
            review.same(item['scores'],previous['raw_scores'],'Literal thirteenth score map')
            review.same(item['evaluated'],previous['evaluated'],'Literal thirteenth evaluated mask')
            observed = review.array(item['evaluated'],shape=(384,384),dtype=np.bool_)
            scores = review.array(item['scores'],shape=(384,384),dtype=np.float32)
            require(observed[mask].all() and np.isfinite(scores[mask]).all() and
                    np.all((scores[mask]>=0)&(scores[mask]<=1)),'Fixed diagnostic support unscored/invalid; no shrinking')
            value = ranking(labels[mask],scores[mask]); actual = output['models'][role]
            for key in ('auc','ap'):
                review.close(actual[key],value[key],'ranking/'+name+'/'+role+'/'+key)
                review.close(previous[key],value[key],'unchanged-ranking/'+name+'/'+role+'/'+key)
                ranking_errors.append(abs(actual[key]-previous[key]))
            operations = {rule:operating(labels[mask],scores[mask],threshold) for rule,threshold in thresholds[role].items()}
            review.tree(actual['operating'],operations,'operating/'+name+'/'+role)
            review.tree(operations['common_baseline'],previous['fixed_baseline_operating'],'unchanged-common/'+name+'/'+role)
            review.same(actual['raw_scores'],item['scores'],'Reported diagnostic score pin')
            review.same(actual['evaluated'],item['evaluated'],'Reported diagnostic coverage pin')
            require(actual['ranking_integrity_max_abs_error']==max(abs(actual[k]-previous[k]) for k in ('auc','ap')),
                    'Reported ranking integrity error differs')
            models[role] = dict(auc=value['auc'],ap=value['ap'],operating=operations)
        for role in ROLES:
            changes = dict(primary_vs_common=delta(models[role]['operating']['primary'],models[role]['operating']['common_baseline']),
                primary_vs_baseline=delta(models[role]['operating']['primary'],models['frozen_seed42']['operating']['primary']))
            review.tree(output['models'][role]['deltas'],changes,'deltas/'+name+'/'+role)
            models[role]['deltas'] = changes
        results[name] = dict(physical_group=case['physical_group'],models=models,counts=dict(zip(('pixels','positive','negative'),DIAGNOSTIC_COUNTS[name])))
    require(set(report['per_context'])==set(DIAGNOSTIC_IDS),'Diagnostic context selection changed')
    groups = {group:{name:row for name,row in results.items() if row['physical_group']==group}
              for group in ('PHerc0814','PHercParis1Fr39')}
    require(all(len(g)==2 for g in groups.values()),'Two contexts per specimen required')
    macros,screens,increases,baseline_floor = {},{a:{} for a in ARMS},{a:{} for a in ARMS},{}
    for group,source in groups.items():
        rows = list(source.values())
        macros[group] = {role:dict(auc=math.fsum(row['models'][role]['auc'] for row in rows)/2,
            ap=math.fsum(row['models'][role]['ap'] for row in rows)/2,
            operating={rule:macro(rows,role,rule) for rule in RULES}) for role in ROLES}
        baseline = {name:row['models']['frozen_seed42']['operating']['primary'] for name,row in source.items()}
        baseline_floor[group] = {name:20*row['confusion']['fp'] <= row['confusion']['fp']+row['confusion']['tn']
                                 for name,row in baseline.items()}
        for arm in ARMS:
            current = {name:row['models'][arm]['operating']['primary'] for name,row in source.items()}
            screens[arm][group] = published_screen(current,baseline)
            differences = {name:exact_sensitivity(row['models'][arm]['operating']['primary'])-
                exact_sensitivity(row['models'][arm]['operating']['common_baseline']) for name,row in source.items()}
            average = sum(differences.values(),Fraction())/2
            increases[arm][group] = dict(equal_context_mean_delta=published_fraction(average),increased=average>0,
                per_context_delta={name:published_fraction(v) for name,v in differences.items()},primary_screen_not_redefined=True)
    review.tree(report['domain_macro'],macros,'domain-macros')
    review.tree(report['screens'],screens,'exact-screens')
    review.tree(report['sensitivity_increased_vs_common'],increases,'exact-increases-vs-common')
    review.tree(report['baseline_specificity_floor'],baseline_floor,'baseline-specificity-floor')
    review.tree(report['both_specimen_screens_passed'],{a:all(r['passed'] for r in screens[a].values()) for a in ARMS},
                'both-specimen-screens')
    require(report['unchanged_ranking_max_abs_error']==max(ranking_errors),'Global unchanged ranking error differs')
    require(sum(r['tiles_verified'] for a in ARMS for r in blend_checks[a].values())==118,'Raw-tile coverage count differs')
    for source_pin in source_pins: review.pin(source_pin)
    review.unchanged()
    return dict(status='PASS',completed_at_utc=datetime.now(timezone.utc).isoformat(),inputs=entries,
        verifier_sources=source_pins,calibration=calibration,blend_reconstruction=blend_checks,normalized_inputs=input_checks,
        per_context=results,domain_macro=macros,screens=screens,sensitivity_increased_vs_common=increases,
        baseline_specificity_floor=baseline_floor,confusion_counts_exact=True,cutoff_float32_bits_exact=True,
        Fraction_screen_arithmetic_exact=True,AUC_AP_unchanged_on_literal_thirteenth_maps=True,
        six_new_blended_maps_verified=True,new_tile_logits_and_normalized_inputs_verified=118,
        scalar_comparisons=review.scalar_comparisons,maximum_scalar_errors=review.maximum_errors,
        verified_input_pins=list(review.files.values()),source_pins_unchanged=True,
        weight_metadata_not_read=list(review.weight_metadata.values()),
        actual_operations=dict(retained_NPY_JSON_reads=True,Torch_imports=0,model_calls=0,checkpoint_reads=0,
                              network_calls=0,new_predictions_or_training=0),
        scope='Independent saved-output math and source/mask/input identity. State/schema/storage/no-gradient checks remain worker receipts; no checkpoint bytes/tensors or model execution by reviewer.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--authorized-retained-read',action='store_true')
    for name in ('plan','locks-ready','metrics'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    require('torch' not in sys.modules,'Torch must not be imported')
    require(HERE in args.output_dir.resolve().parents and not args.output_dir.exists(),'Fresh owned review output directory required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources = [record(p) for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts]
    try:
        require(record(HERE/'blending/blend_verifier.py')['sha256']==BLEND_SOURCE_SHA,'Sealed independent blend helper changed')
        if args.self_test:
            math_checks,mask_checks = numeric_fixtures(),mask_fixtures()
            blend_receipt = json.loads((HERE/'blending/synthetic-fixtures-v2.json').read_text())
            require(blend_receipt['status']=='PASS','Sealed blend synthetic tests did not PASS')
            result = dict(status='PASS',numerical_fixtures=math_checks,mask_fixtures=mask_checks,
                sealed_blend_synthetic_receipt=record(HERE/'blending/synthetic-fixtures-v2.json'),sources=sources,
                actual_retained_array_reads=0,Torch_imports=0,model_calls=0,checkpoint_reads=0,network_calls=0)
        else:
            require(args.authorized_retained_read,'Root authorization after scientific outputs PASS required')
            result = run_review(args,sources)
    except Exception as error:
        result = dict(status='STOPPED',completed_at_utc=datetime.now(timezone.utc).isoformat(),
            reason=str(error),error_type=type(error).__name__,no_partial_PASS_claim=True,verifier_sources=sources,
            model_calls=0,Torch_imports=0,checkpoint_reads=0,network_calls=0)
        (args.output_dir/'independent-verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        raise
    require('torch' not in sys.modules,'Torch must not be imported')
    target = args.output_dir/('synthetic-fixtures.json' if args.self_test else 'independent-verification.json')
    target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',receipt=record(target))))


if __name__=='__main__':main()
