"""Independent saved-score retention metrics; no models, weights or network."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import numpy as np
from independent_metrics import compute


def offline(*args, **kwargs):
    raise RuntimeError('Network disabled in independent retention scoring')


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline
HERE = Path(__file__).resolve().parent
ARMS = ['A_preserve', 'B_preserve']
CASES = ['pherc0814-46527-fresh-rank01', 'pherc0814-46527-fresh-rank14',
         'frag4-context1', 'frag4-context2']
COUNTS = {CASES[0]: (49033,19545,29488), CASES[1]: (21328,4767,16561),
          CASES[2]: (91444,3818,87626), CASES[3]: (77851,15252,62599)}
CUTOFF = 0.7558947205543518
DESIGN_SHA = '59bee53d3f1a662b37aec43fde31698ce0a9946ea0dd0de7027c01c3cf254f36'
METRIC_SHA = '08976bcbc229fa83175c6dd7d4f569704ddbd494f47efec97a1e0e1e9098e119'


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def identity(item):
    sizes = [item[k] for k in ['size_bytes','bytes','byte_length'] if k in item]
    require(sizes and all(type(n) is int and n >= 0 and n == sizes[0] for n in sizes), 'Invalid byte-length pin')
    digest = item['sha256']
    require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'Invalid SHA pin')
    require(Path(item['path']).is_absolute(), 'Relative artifact path')
    return dict(path=str(Path(item['path']).resolve()),size_bytes=sizes[0],sha256=digest)


class Pins:
    def __init__(self):
        self.files = {}
        self.weight_metadata = {}

    def check(self, item):
        expected = identity(item)
        path = expected['path']
        table = self.weight_metadata if Path(path).suffix in ['.pth','.pt','.ckpt','.safetensors'] else self.files
        if path in table:
            require(table[path] == expected, 'Conflicting pin: ' + Path(path).name)
        else:
            if table is self.files:
                require(record(path)==expected, 'File pin changed: ' + Path(path).name)
            table[path] = expected
        return Path(path)

    def collect(self, value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value and any(k in value for k in ['size_bytes','bytes','byte_length']):
                self.check(value)
            else:
                for child in value.values():
                    self.collect(child)
        elif isinstance(value,list):
            for child in value:
                self.collect(child)

    def json(self, item):
        value = json.loads(self.check(item).read_text())
        self.collect(value)
        return value

    def unchanged(self):
        for item in self.files.values():
            require(record(item['path'])==item, 'A file changed during review: ' + Path(item['path']).name)


def same(first, second, message):
    require(identity(first)==identity(second),message)


def close(first, second, name):
    require(type(first) in [int,float] and type(second) in [int,float]
            and math.isfinite(first) and math.isfinite(second)
            and math.isclose(first,second,rel_tol=0,abs_tol=1e-12), 'Metric disagreement: '+name)


def tree(expected, actual, name):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(expected)==set(actual), 'Metric keys differ: '+name)
        for key,value in expected.items():
            tree(value,actual[key],name+'/'+key)
    elif isinstance(expected,bool):
        require(actual is expected, 'Boolean screen differs: '+name)
    else:
        close(expected,actual,name)


def array(pins, item, dtype, shape):
    value=np.load(pins.check(item),allow_pickle=False)
    require(value.dtype==np.dtype(dtype) and value.shape==shape,'Saved array dtype/shape differs: '+Path(item['path']).name)
    return value


def labels_and_primary(pins, case):
    legacy=case['legacy_case']
    if case['family']=='pooled21':
        require(case['id'] in CASES[:2] and case['physical_group']=='PHerc0814'
                and case['layer_start']==2 and legacy['evaluation_participation_role']=='supplied_supervision',
                'Pooled evaluation role/depth differs')
        source=legacy['inputs']
        labels=array(pins,source['inklabels'],'uint8',(384,384))
        observed=array(pins,source['acquisition_observed_mask'],'bool',(384,384))
        require(observed.all(),'Original acquired support is incomplete')
        supervision=np.load(pins.check(source['supervision_mask']),allow_pickle=False)
        participation=array(pins,source['evaluation_participation_mask'],'bool',(384,384))
        require(supervision.shape==(384,384) and supervision.dtype in [np.dtype('uint8'),np.dtype('bool')]
                and np.all((supervision==0)|(supervision==1)) and np.array_equal(participation,supervision>0),
                'Supplied-supervision participation differs')
        border=np.zeros((384,384),dtype=bool);border[64:320,64:320]=True
        primary=(supervision>0)&observed&border
        same(case['primary_mask'],legacy['data']['arrays']['primary_annotation_mask'], 'Original PHerc0814 mask pin differs')
    else:
        require(case['family']=='frag4_float17' and case['id'] in CASES[2:]
                and case['physical_group']=='PHercParis1Fr39' and case['layer_start']==0,
                'Frag4 evaluation family/depth differs')
        source=legacy['arrays']
        labels=np.load(pins.check(source['labels']),allow_pickle=False)
        require(labels.shape==(384,384) and labels.dtype in [np.dtype('uint8'),np.dtype('bool')], 'Frag4 label encoding differs')
        known=array(pins,source['known_mask'],'bool',(384,384))
        surface_mask=array(pins,source['surface_mask'],'bool',(384,384))
        require(np.all(~known|surface_mask), 'Known Frag4 participation escapes exposed-surface support')
        border=np.zeros((384,384),dtype=bool);border[32:352,32:352]=True
        primary=known&border
        same(case['primary_mask'],source['primary_mask'], 'Original Frag4 mask pin differs')
    require(np.all((labels==0)|(labels==1)), 'Supplied reference is not binary')
    labels=labels.astype(bool)
    saved=array(pins,case['primary_mask'],'bool',(384,384))
    require(np.array_equal(saved,primary),'Canonical primary differs from original support and margin')
    total,positive,negative=COUNTS[case['id']]
    require(int(primary.sum())==total and int((primary&labels).sum())==positive
            and int((primary&~labels).sum())==negative, 'Frozen primary class counts differ')
    require(case['canonical_counts']==dict(fixed_primary_count=total,positive=positive,negative=negative),
            'Plan class-count witness differs')
    return labels,primary


def captured(pins,item,case):
    grid=[[x,y,128,128] for y in range(0,257,64) for x in range(0,257,64)]
    meta=item['inference']
    require(meta['tile_count']==meta['scored_tile_count']==25
            and [t['bbox_roi_xywh'] for t in meta['tiles']]==grid
            and all(t['status']=='scored' for t in meta['tiles']), 'Fixed full tile grid changed')
    tiles=item['captured_tiles']
    require(len(tiles)==25 and [t['bbox_roi_xywh'] for t in tiles]==grid, 'Captured tile order differs')
    for tile in tiles:
        digest=tile['input_sha256']
        require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'Malformed normalized input fingerprint')
    logits=array(pins,item['logits'],'float32',(25,128,128))
    require(np.isfinite(logits).all(),'Raw logits nonfinite')
    require(item['adapter_layer_start']==case['layer_start'] and item['adapter_direction']=='forward'
            and item['model_input_tensor_unchanged'] is True, 'Adapter direction/input contract differs')
    baseline=case['historical_maps']['frozen_seed42']
    require([t['bbox_roi_xywh'] for t in baseline['inference']['tiles'] if t['status']=='scored']==grid,
            'Retained baseline grid differs')
    if 'captured_tiles' in baseline:
        require(tiles==baseline['captured_tiles'] and item['archived_normalized_input_fingerprint_match'] is True,
                'Captured Frag4 normalized input differs from retained baseline')
    else:
        require(item['archived_normalized_input_fingerprint_match'] is None
                and isinstance(item['archived_input_limit'],str) and item['archived_input_limit'],
                'Missing legacy fingerprint limit was not disclosed')
    return tiles


def verify(plan_path,run_path,metrics_path):
    pins=Pins()
    own=[record(HERE/'verify_retention.py'),record(HERE/'independent_metrics.py')]
    require(own[1]['sha256']==METRIC_SHA,'Previously tested independent metrics changed')
    for item in own:pins.check(item)
    plan_record=record(plan_path);plan=pins.json(plan_record)
    require(plan['kind']=='conditional-final-state-retention-v1' and plan['ready_for_execution'] is True,
            'Retention plan is not ready')
    require(plan['arms']==ARMS and [c['id'] for c in plan['cases']]==CASES, 'Fixed context/arm scope differs')
    require(plan['eligible_arms']==[a for a in ARMS if a in plan['models']], 'Qualified arm order differs')
    eligible=plan['eligible_arms']
    require(plan['limits']['maximum_tile_forwards']==200
            and plan['limits']['maximum_tile_forwards_per_arm']==100
            and plan['limits']['retries']==0, 'Inference budget differs')
    require(plan['inference']['direction']=='forward' and plan['inference']['mask_passed_to_adapter'] is False
            and plan['inference']['no_baseline_forward'] is True
            and plan['inference']['no_training'] is True and plan['inference']['no_threshold_fitting'] is True,
            'Inference scope changed')
    require(plan['baseline_cutoff']['value']==CUTOFF, 'Common frozen baseline cutoff differs')
    lock=pins.json(plan['baseline_cutoff']['lock'])
    require(lock['primary']==CUTOFF and lock['arm']=='frozen_seed42', 'Original cutoff lineage differs')
    for case in plan['cases']:
        require(case['fixed_baseline_cutoff']==CUTOFF, 'Case cutoff differs')
    gate=pins.json(plan['training_gate'])
    same(gate['plan'],plan['training_plan'],'Gate was derived from a different training plan')
    require(gate['status']=='PASS' and gate['gate_contract']==plan['gate_contract']
            and gate['eligible_arms']==eligible and set(gate['arms'])==set(ARMS), 'Training-only gate provenance differs')
    require(plan['gate_contract']['minimum_improvement']==1e-4
            and plan['gate_contract']['base_and_full_required'] is True
            and plan['gate_contract']['fixed_exposures']==32, 'Training eligibility criterion differs')
    for arm in ARMS:
        entry=gate['arms'][arm];values=entry['fixed_evaluation']
        require(values['exposures']==32 and all(type(values[key]) in [int,float] and math.isfinite(values[key])
            for key in ['base_initial','base_final','full_initial','full_final']), 'Gate losses nonfinite or incomplete')
        base_delta=values['base_final']-values['base_initial']
        full_delta=values['full_final']-values['full_initial']
        require(values['base_delta']==base_delta and values['full_delta']==full_delta,
                'Gate final-minus-initial arithmetic differs')
        expected=base_delta<=-1e-4 and full_delta<=-1e-4
        require(entry['eligible'] is expected and expected==(arm in eligible), 'Training-only eligibility disagrees')
        training=pins.json(entry['training_result'])
        same(training['plan'],plan['training_plan'],'Training source plan differs')
        require(training['status']=='PASS' and training['arm']==arm
                and training['initial_state_sha256']==plan['initial_state_sha256']
                and training['eligible'] is expected
                and training['actual_calls']==dict(student_forwards=192,teacher_forwards=0,backwards=32,optimizer_steps=32),
                'Actual training receipt/accounting differs')
        same(training['checkpoint'],entry['checkpoint'],'Training checkpoint metadata differs')
        require(training['final_state_sha256']==entry['final_state_sha256']
                and training['fixed_evaluation']['32']['state_sha256']==entry['final_state_sha256'],
                'Final fixed-risk state differs')
        for suffix,risk in [('initial',training['baseline_fixed_evaluation']),('final',training['fixed_evaluation']['32'])]:
            require(len(risk['per_batch'])==32,'Fixed risk lacks 32 equal batch exposures')
            for objective in ['base','full']:
                batches=[row[objective] for row in risk['per_batch']]
                require(all(type(value) in [int,float] and math.isfinite(value) for value in batches),
                        'A fixed-batch loss is nonfinite')
                average=math.fsum(batches)/32
                require(risk['equal_batch_means'][objective]==average
                        and values[objective+'_'+suffix]==average,'Gate mean is not the exact fixed batch reduction')
        if expected:
            source=plan['models'][arm]
            require(source['role']==arm and source['learning_rate']==1e-6
                    and source['fixed_evaluation']==values
                    and source['state_sha256']==entry['final_state_sha256'], 'Qualified final state/risk binding differs')
            same(source['checkpoint'],entry['checkpoint'],'Qualified checkpoint metadata differs')
            same(source['training_result'],entry['training_result'],'Training result pin differs')
            require(training['final_state_sha256']==source['state_sha256'], 'Trained state receipt differs')
            same(training['checkpoint'],source['checkpoint'],'Saved checkpoint receipt differs')
    design=HERE.parent/'RESEARCH_DESIGN.md'
    require(record(design)['sha256']==DESIGN_SHA,'Frozen scientific retention design changed')
    pins.check(record(design))
    for role in ['frozen_seed42']+ARMS:
        historical=plan['historical'][role]
        pooled=pins.json(historical['pooled_result'])
        frag=pins.json(historical['frag4_result'])
        require(pooled['status']==frag['status']=='PASS' and pooled['arm']==role and frag['role']==role,
                'Historical role result lineage differs')
        old_state=pooled['initial_state_sha256'] if role=='frozen_seed42' else pooled['training']['final_state_sha256']
        require(old_state==historical['state_sha256']==frag['state_sha256'], 'Reused specimen maps have different states')
        require(historical['learning_rate'] is None if role=='frozen_seed42' else historical['learning_rate']==1e-4,
                'Matched old learning-rate metadata differs')
        for case in plan['cases']:
            archived=pooled.get('diagnostic',pooled.get('fresh',{}))[case['id']] if case['family']=='pooled21' else frag['runs']['forward'][case['id']]
            require(case['historical_maps'][role]==archived, 'Reused map was substituted or assigned to another role')
    index_record=record(run_path);index=pins.json(index_record)
    report_record=record(metrics_path);report=pins.json(report_record)
    same(index['plan'],plan_record,'Runner plan pin differs');same(report['plan'],plan_record,'Analyzer plan pin differs')
    same(report['run_index'],index_record,'Analysis points to a different run')
    require(index['eligible_arms']==report['eligible_arms']==eligible and set(index['models'])==set(eligible)
            and index['retries']==0, 'Conditional run coverage differs')
    if not eligible:
        require(index['status']==report['status']=='SKIPPED' and index['models']=={}
                and index['tile_forward_count']==0 and report['per_context']=={} and report['screen'] is None,
                'No-eligible branch performed or claimed retention')
        pins.unchanged()
        return dict(status='SKIPPED_VERIFIED',plan=plan_record,run_index=index_record,analysis=report_record,
            eligible_arms=[],new_runs=0,new_tile_forwards=0,model_calls=0,network_calls=0,checkpoint_reads=0,
            reason='No arm qualified from the fixed training-risk gate; no substitute or baseline inference.')
    require(index['status']==report['status']=='PASS' and report['model_calls']==report['threshold_fitting_calls']==0,
            'Conditional retention/analysis incomplete')
    require(index['workers']==[dict(arm=a,returncode=0) for a in eligible], 'Worker coverage/exit status differs')
    require(index['complete_fixed_contexts_per_eligible_state'] is True, 'Incomplete fixed panel')
    workers={arm:pins.json(item) for arm,item in index['models'].items()}
    for arm,worker in workers.items():
        require(worker['status']=='PASS' and worker['arm']==arm and worker['plan']==plan_record
                and worker['source_model']==plan['models'][arm]
                and worker['state_sha256']==plan['models'][arm]['state_sha256']
                and set(worker['runs'])==set(CASES) and worker['tile_forward_count']==100
                and worker['training_calls']==worker['threshold_fitting_calls']==worker['baseline_forward_calls']==0
                and worker['model_state_unchanged'] is True and worker['network_requests']==0,
                'New worker state/accounting differs')
    roles=['frozen_seed42','old_A_preserve_1e4','old_B_preserve_1e4']+eligible
    require(report['roles']==roles and set(report['per_context'])==set(CASES)
            and report['baseline_cutoff']==plan['baseline_cutoff'] and report['model_results']==index['models']
            and report['retained_models']==plan['historical'],
            'Reported role/cutoff/result pins differ')
    results={};total_tiles=0
    for case in plan['cases']:
        name=case['id'];labels,mask=labels_and_primary(pins,case)
        row=report['per_context'][name]
        require(row['physical_group']==case['physical_group'] and row['input_family']==case['family']
                and row['layer_start']==case['layer_start'] and set(row['models'])==set(roles), 'Reported context role differs')
        fixed=array(pins,row['fixed_mask'],'bool',(384,384))
        require(np.array_equal(mask,fixed) and (row['pixel_count'],row['positive'],row['negative'])==COUNTS[name],
                'Analysis primary support changed')
        maps=dict(frozen_seed42=case['historical_maps']['frozen_seed42'],
            old_A_preserve_1e4=case['historical_maps']['A_preserve'],
            old_B_preserve_1e4=case['historical_maps']['B_preserve'])
        fingerprint=None
        for arm,worker in workers.items():
            item=worker['runs'][name];tiles=captured(pins,item,case);total_tiles+=len(tiles)
            if fingerprint is None:fingerprint=tiles
            else:require(fingerprint==tiles,'New arms saw different normalized inputs')
            maps[arm]=item
        metrics={}
        for role,item in maps.items():
            if 'captured_tiles' in case['historical_maps']['frozen_seed42']:
                require(item['captured_tiles']==case['historical_maps']['frozen_seed42']['captured_tiles'],
                        'Retained or new Frag4 inputs differ across roles')
            if role in eligible:
                require(item['normalization']==plan['inference']['normalization'],
                        'New worker normalization contract differs')
            observed=array(pins,item['evaluated'],'bool',(384,384))
            scores=array(pins,item['scores'],'float32',(384,384))
            require(observed[mask].all() and np.isfinite(scores[mask]).all()
                    and np.all((scores[mask]>=0)&(scores[mask]<=1)), 'A fixed primary pixel is unscored/nonfinite')
            same(row['models'][role]['raw_scores'],item['scores'],'Reported raw score pin differs')
            same(row['models'][role]['evaluated'],item['evaluated'],'Reported evaluated-mask pin differs')
            value=compute(labels[mask],scores[mask],{'fixed_baseline':CUTOFF})
            actual=row['models'][role]
            close(value['auc'],actual['auc'],'AUC/'+name+'/'+role);close(value['ap'],actual['ap'],'AP/'+name+'/'+role)
            expected=value['operating']['fixed_baseline'];operating=actual['fixed_baseline_operating']
            require(operating['threshold']==CUTOFF
                    and operating['confusion']=={key:expected[key] for key in ['tp','fp','tn','fn']}
                    and operating['pixel_count']==value['pixels']
                    and operating['supplied_positive_pixels']==value['positive']
                    and operating['supplied_negative_pixels']==value['negative'],'Frozen-cutoff counts differ')
            for key in ['sensitivity','specificity']:close(operating[key],expected[key],key+'/'+name+'/'+role)
            metrics[role]=dict(auc=value['auc'],ap=value['ap'],sensitivity=expected['sensitivity'],specificity=expected['specificity'])
        differences={a:{comparison:{key:metrics[a][key]-metrics[comparison][key] for key in ['auc','ap']}
                       for comparison in ['frozen_seed42','old_'+a+'_1e4']} for a in eligible}
        tree(differences,row['deltas'],'Percontext baseline/matched deltas')
        results[name]=dict(physical_group=case['physical_group'],counts=dict(zip(['pixels','positive','negative'],COUNTS[name])),
                          models=metrics,deltas=differences)
    require(total_tiles==100*len(eligible)==index['tile_forward_count']==report['tile_forward_count']<=200,
            'Composite new-run accounting differs')
    groups={group:[r for r in results.values() if r['physical_group']==group] for group in ['PHerc0814','PHercParis1Fr39']}
    require(all(len(rows)==2 for rows in groups.values()),'Two-specimen grouping differs')
    def mean(rows):
        return {role:{key:sum(row['models'][role][key] for row in rows)/len(rows)
                      for key in ['auc','ap','sensitivity','specificity']} for role in roles}
    domain_macro={group:mean(rows) for group,rows in groups.items()}
    context_macro=mean(list(results.values()))
    specimen_macro={role:{key:sum(domain_macro[g][role][key] for g in groups)/2
                         for key in ['auc','ap','sensitivity','specificity']} for role in roles}
    tree(domain_macro,report['domain_macro'],'Domain macro');tree(context_macro,report['equal_context_macro'],'Equalcontext macro')
    tree(specimen_macro,report['equal_specimen_macro'],'Equalspecimen macro')
    for key in ['domain_macro_auc_delta_min','domain_macro_ap_delta_min']:
        require(plan['screen_contract'][key]==-.01,'Frozen domain retention tolerance differs')
    for key in ['per_context_auc_delta_min','per_context_ap_delta_min']:
        require(plan['screen_contract'][key]==-.02,'Frozen percontext retention tolerance differs')
    screens={}
    for arm in eligible:
        contexts={name:{key:row['models'][arm][key]-row['models']['frozen_seed42'][key]>=-.02 for key in ['auc','ap']}
                  for name,row in results.items()}
        domains={group:{key:row[arm][key]-row['frozen_seed42'][key]>=-.01 for key in ['auc','ap']}
                 for group,row in domain_macro.items()}
        passed=all(all(check.values()) for check in list(contexts.values())+list(domains.values()))
        reported=report['screen'][arm]
        tree(contexts,reported['context_checks'],'Retention context screen')
        tree(domains,reported['domain_macro_checks'],'Retention domain screen')
        require(reported['passed'] is passed and reported['automatic_promotion'] is False,'Retention screen/promotion differs')
        screens[arm]=dict(context_checks=contexts,domain_macro_checks=domains,passed=passed,automatic_promotion=False)
    pins.unchanged()
    return dict(status='PASS',completed_at_utc=datetime.now(timezone.utc).isoformat(),plan=plan_record,
        run_index=index_record,analysis=report_record,verifier_sources=own,
        verified_file_pins=len(pins.files),file_pins_unchanged_before_after=True,
        checkpoint_metadata_pins=list(pins.weight_metadata.values()),checkpoint_file_reads=0,
        eligible_arms=eligible,new_run_count=4*len(eligible),new_tile_forward_count=total_tiles,
        per_context=results,domain_macro=domain_macro,equal_context_macro=context_macro,
        equal_specimen_macro=specimen_macro,screen=screens,
        all_fixed_primary_pixels_evaluated_and_finite=True,confusion_counts_exact=True,
        metric_numeric_absolute_tolerance=1e-12,model_calls=0,network_calls=0,checkpoint_deserializations=0,
        source_surface_array_decodes=0,
        limits=['Semantic state and checkpoint identities checked as pinned metadata/worker receipts; checkpoint bytes and tensors are not independently read.',
            'Raw logit shape/placement/finiteness checked; model forwards and numerical sigmoid/Hann blending not replayed.',
            'Legacy PHerc0814 lacks archived normalized-input fingerprints; identical inputs are bounded by frozen source/depth/normalizer pins and reported placement.',
            'Four reused contexts are two specimens and selected training-risk states; no prospective or historically independent accuracy, significance, novelty or reading claim.'],
        verified_inputs=list(pins.files.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','results','metrics','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();output=args.output_dir.resolve()
    require(HERE in output.parents and not output.exists(),'Choose a new receipt directory under retention-review')
    output.mkdir(parents=True)
    try:
        receipt=verify(args.plan,args.results,args.metrics)
    except Exception as error:
        receipt=dict(status='STOPPED',reason=str(error),error_type=type(error).__name__,model_calls=0,
                     network_calls=0,checkpoint_reads=0,no_partial_verification_claim=True)
        (output/'verification.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        raise
    target=output/'verification.json'
    target.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],receipt=record(target))))


if __name__=='__main__':main()
