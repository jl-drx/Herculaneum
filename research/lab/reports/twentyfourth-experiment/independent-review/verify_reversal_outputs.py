"""Independent paired depth-permutation verification from saved evidence only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import time
import numpy as np
from review_core import (HERE,PHASE,IDS,PINS,need,pin,checked,json_read,inherited,load_array,
 verify_context_input,raw_metric,metric_delta,verify_arrays,forward_arrays,seed42_value_hashes,
 compare_model_identity,compare_forward_reference)

PROTOCOL_SHA='f9cc08b99144f55e5842b94e08cb92a8524badafd58d5bf7483254953641c3ee'

def parent_metrics(contexts,parent):
    ids=[cid for cid in IDS if contexts[cid]['parent_papyrus']==parent]
    need(len(ids)=={'Paris1':2,'Paris2':4}[parent],'parent_context_count')
    models={}
    for role in ('seed42','native'):
        value={direction:{key:float(np.mean([contexts[cid]['models'][role][direction][key] for cid in ids]))
            for key in ('roc_auc','average_precision')} for direction in ('forward','reverse')}
        value['reverse_minus_forward']=metric_delta(value['reverse'],value['forward']);models[role]=value
    return dict(context_ids=ids,models=models,difference_of_changes=metric_delta(
        models['native']['reverse_minus_forward'],models['seed42']['reverse_minus_forward']))

def verify(plan_path,run_path,metrics_path,release_path):
    run=Path(run_path);plan_path=Path(plan_path)
    freeze=json.loads((HERE/'CHECKER_FREEZE.json').read_text())
    need(freeze['author_plan']==pin(plan_path),'checker_author_plan')
    for record in freeze['code'].values():checked(record)
    need(pin(PHASE/'PROTOCOL.md')['sha256']==PROTOCOL_SHA,'protocol_pin')
    plan=json.loads(plan_path.read_text());release=json.loads(Path(release_path).read_text())
    need(release['status']=='GO' and release['plan']==pin(plan_path) and release['code_freeze']==plan['code_freeze'],'root_release')
    for record in plan['sources'].values():checked(record)
    for record in plan['code'].values():checked(record)
    need(json_read(plan['code_freeze'])['code']==plan['code'],'author_code_freeze')
    contract=json_read(plan['sources']['input_contract'])
    need(contract['status']=='PASS_NORMALIZED_TENSOR_DEPTH_REVERSAL_CONTRACT','canonical_input_contract')
    need([case['id'] for case in contract['cases']]==IDS,'case_order')
    policy=contract['input_policy']
    need(policy['adapter_read_direction']=='forward' and policy['model_depth_indices']==list(range(16,-1,-1))
        and policy['tensor_axis']==2 and policy['no_raw_input_depth_reversal_in_execution'] is True
        and policy['no_repeated_normalization'] is True,'actual_input_operation')
    maths,evidence,helper=inherited()
    phase22=json_read(plan['sources']['phase22_plan']);schema=json_read(phase22['sources']['reference_schema'])
    native_hashes=helper.qualified_hashes(phase22['stage1']['artifact'],schema)
    seed42=seed42_value_hashes(plan['seed42_checkpoint'],'model',schema)
    need(seed42['legacy_semantic_sha256']==plan['seed42_state_sha256']==contract['backbone_state_sha256'],'seed42_legacy_state')
    need(seed42['config']==json_read(plan['sources']['seed42_bound_config']) and seed42['step']==74999,'seed42_original_config_step')
    prior_native=json_read(contract['native_load_verification'])
    helper.verify_state(prior_native,native_hashes,schema)
    controller=json.loads((run/'controller-result.json').read_text())
    need(controller['status']=='PASS' and controller['plan']==pin(plan_path) and controller['no_retry'] is True,'controller_binding')
    need(controller['model_constructions']==controller['strict_loads']==2 and controller['full_model_calls']==300
        and controller['elapsed_seconds']<=420,'controller_limits_counts')
    need([w['model'] for w in controller['workers']]==['seed42','native'] and set(controller['results'])=={'seed42','native'},'worker_order')
    results={}
    for role,worker in zip(('seed42','native'),controller['workers'],strict=True):
        need(worker['returncode']==0 and worker['stop'] is None and worker['external_peak_rss_bytes']<=4*1024**3,'external_worker_result')
        result=json_read(controller['results'][role]);results[role]=result
        need(result['status']=='PASS' and result['model']==role and result['direction']=='reverse' and result['plan']==pin(plan_path)
            and result['input_contract']==plan['sources']['input_contract'],'worker_binding')
        need(result['model_constructions']==result['strict_loads']==1 and result['full_model_calls']==150
            and result['training_steps']==result['gradients']==0,'worker_counts')
        need(result['source_hashes_unchanged'] is True and result['state_unchanged'] is True and all(v==0 for v in result['forbidden_calls'].values()),'worker_invariants')
        need(result['elapsed_seconds']<=180 and result['cpu_seconds']<=360 and result['max_rss_bytes']<=4*1024**3,'worker_limits')
        expected_loads={'torch_weights_only':int(role=='seed42'),'safetensors':int(role=='native')}
        need(result['safe_load_counts']==expected_loads,'safe_load_counts')
        initial=json_read(result['load_verification']);final=json_read(result['final_state_verification'])
        need(initial['strict'] is True and initial['missing_keys']==[] and initial['unexpected_keys']==[]
            and initial['safe_load_counts']==expected_loads,'strict_load')
        if role=='seed42':
            compare_model_identity(role,initial['model'],initial['checkpoint'],plan['seed42_checkpoint'],initial['legacy_state_sha256'],plan['seed42_state_sha256'])
            need(initial['container']=='model' and initial['config']==seed42['config'],'seed42_container_config')
            hashes=seed42['hashes']
        else:
            compare_model_identity(role,initial['model'],initial['checkpoint'],phase22['stage1']['artifact'],initial['semantic_value_sha256'],prior_native['semantic_value_sha256'])
            need(initial['container']=='bare_safetensors' and initial['config']==prior_native['config'],'native_container_config')
            hashes=native_hashes
        helper.verify_state(initial,hashes,schema);helper.verify_state(final,hashes,schema)
        need(set(result['runs'])==set(IDS),'worker_cases')
    published=json.loads(Path(metrics_path).read_text())
    need(published['status']=='PASS' and published['plan']==pin(plan_path)
        and published['controller_result']==pin(run/'controller-result.json')
        and published['input_contract']==plan['sources']['input_contract'],'analysis_binding')
    need(published['model_calls']==published['training_steps']==published['threshold_fits']==published['threshold_classifications']==0
        and published['automatic_promotion'] is False and published['direction_selection'] is False
        and published['performance_pass_gate'] is None,'analysis_scope')
    need(published['elapsed_seconds']<=120 and published['max_rss_bytes']<=4*1024**3,'analysis_limits')
    contexts={};arithmetic={};counts={};input_checks=0
    for case in contract['cases']:
        cid=case['id'];source,forward_hashes,reverse_hashes=verify_context_input(case,helper.normalize,helper)
        models={};arithmetic[cid]={};score_inputs={}
        counts[cid]={k:case['counts'][k] for k in ('primary','primary_positive','primary_negative')}
        for role in ('seed42','native'):
            meta=results[role]['runs'][cid]
            need(meta==json.loads((run/role/cid/'metadata.json').read_text()),'case_metadata_binding')
            need(meta['status']=='PASS' and meta['model']==role and meta['id']==cid and meta['direction']=='reverse'
                and meta['adapter_read_direction']=='forward' and meta['model_depth_permutation']==list(range(16,-1,-1)),'case_intervention')
            need(meta['input_contract']==plan['sources']['input_contract'] and meta['surface']==case['arrays']['surface']
                and meta['primary_mask']==case['arrays']['primary_mask'],'case_input_bindings')
            need(meta['original_captured_tiles']==forward_hashes and meta['captured_tiles']==reverse_hashes
                and meta['input_fingerprints_exact_before_forward'] is True and meta['surface_unchanged'] is True,'exact_input_pairs')
            inf=meta['inference'];need(inf['tile_count']==inf['scored_tile_count']==25 and all(t['status']=='scored' for t in inf['tiles']),'tile_calls')
            need([t['bbox_roi_xywh'] for t in inf['tiles']]==[t['bbox_roi_xywh'] for t in forward_hashes],'tile_schedule')
            need(meta['primary_pixels']==meta['primary_evaluated_pixels']==case['counts']['primary'],'exact_primary_count')
            reverse={key:load_array(record) for key,record in meta['arrays'].items()}
            forward,forward_check=forward_arrays(case['forward_outputs'][role],evidence,helper)
            arithmetic[cid][role]=dict(reverse=verify_arrays(reverse,evidence,helper),forward=forward_check)
            mask=source['primary_mask'];labels=source['labels']
            helper.exact(reverse['evaluated_mask'],forward['evaluated'],'paired_evaluated_mask')
            mf=raw_metric(labels,forward['scores'],mask,forward['evaluated'],maths)
            mr=raw_metric(labels,reverse['raw_scores'],mask,reverse['evaluated_mask'],maths)
            models[role]=dict(forward=mf,reverse=mr,reverse_minus_forward=metric_delta(mr,mf),
                probability_drift=maths.disagreement(reverse['raw_scores'],forward['scores'],mask))
            score_inputs[role]=dict(forward=case['forward_outputs'][role]['scores'],reverse=meta['arrays']['raw_scores'])
            input_checks+=len(forward_hashes)
        contexts[cid]=dict(parent_papyrus=case['parent_papyrus'],fragment=case['fragment'],counts=counts[cid],models=models,
            difference_of_changes=metric_delta(models['native']['reverse_minus_forward'],models['seed42']['reverse_minus_forward']))
        maths.scalar_close(published['per_context'][cid],contexts[cid],cid)
        need(published['per_context'][cid]['primary_mask']==case['arrays']['primary_mask']
            and published['per_context'][cid]['labels']==case['arrays']['labels']
            and published['per_context'][cid]['full_original_primary_support_preserved'] is True,'published_original_support')
        need(published['per_context'][cid]['score_inputs']==score_inputs,'published_score_bindings')
        for role in ('seed42','native'):
            compare_forward_reference(published['per_context'][cid]['score_inputs'][role]['forward'],case['forward_outputs'][role]['scores'],role)
    need(input_checks==300 and sum(v['primary'] for v in counts.values())==520794
        and sum(v['primary_positive'] for v in counts.values())==86535 and sum(v['primary_negative'] for v in counts.values())==434259,'aggregate_counts')
    need(set(published['per_context'])==set(IDS),'published_all_contexts')
    parents={p:parent_metrics(contexts,p) for p in ('Paris1','Paris2')}
    for parent,value in parents.items():maths.scalar_close(published['per_parent'][parent],value,parent)
    for record in plan['sources'].values():checked(record)
    for record in plan['code'].values():checked(record)
    for case in contract['cases']:
        for record in case['arrays'].values():checked(record)
        for role in ('seed42','native'):
            for key in ('scores','logits','evaluated','metadata','result'):checked(case['forward_outputs'][role][key])
    vendor=json_read(phase22['sources']['current_vendor_manifest']);root=Path(phase22['sources']['current_vendor_manifest']['path']).parent
    for record in vendor['files']:need(pin(root/record['path'])['sha256']==record['sha256'],'vendor_final_pin')
    phase_bytes=sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file());need(phase_bytes<=256*1024**2,'phase_output_limit')
    return dict(status='PASS',author_plan=pin(plan_path),root_release=pin(release_path),metrics=pin(metrics_path),
        controller_result=pin(run/'controller-result.json'),per_context=contexts,per_parent=parents,arithmetic=arithmetic,
        full_model_calls_verified=300,normalized_input_pairs_checked=300,unique_input_pairs=150,
        state_hashes_checked=1016,alias_pairs_checked=488,seed42_legacy_state_sha256=seed42['legacy_semantic_sha256'],
        native_prior_semantic_value_sha256=prior_native['semantic_value_sha256'],original_primary_pixels=520794,
        primary_positive=86535,primary_negative=434259,verifier_model_calls=0,network_calls=0,
        phase_bytes_before_receipt=phase_bytes,metric_tolerance=1e-12,
        scope='Paired normalized depth-order sensitivity on reused development; no direction selection or performance gate.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True)
    p.add_argument('--metrics',required=True);p.add_argument('--root-release',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();signal.alarm(120);resource.setrlimit(resource.RLIMIT_CPU,(120,)*2);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
    def deny(event,args):
        if event.startswith('socket.'):raise ValueError('network_forbidden')
    sys.addaudithook(deny);start=time.monotonic();cpu=time.process_time()
    try:result=verify(a.plan,a.run,a.metrics,a.root_release)
    except Exception as exc:
        result=dict(status='STOPPED',failure_type=type(exc).__name__,failure_code=str(exc) if isinstance(exc,ValueError) else 'unexpected_checker_exception')
    result.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    if result['max_rss_bytes']>4*1024**3:result.update(status='STOPPED',failure_code='checker_rss_limit')
    with Path(a.output).open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],receipt=pin(a.output))))
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
