"""Frozen saved-output verification; native/seed42 factories are never imported."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import time
import numpy as np
from review_math import Reject,need,scalar_close
from review_evidence import (HERE,PHASE,IDS,PROTOCOL_SHA,pin,checked,read_json,load_array,
 reference_helpers,load_context_sources,check_arrays,context_metrics,parent_metrics)

def verify(plan_path,run_path,metrics_path,release_path):
    plan_path=Path(plan_path);run=Path(run_path)
    freeze=json.loads((HERE/'CHECKER_FREEZE.json').read_text())
    need(freeze['author_plan']==pin(plan_path),'checker_plan_binding')
    for record in freeze['code'].values():checked(record)
    need(pin(PHASE/'PROTOCOL.md')['sha256']==PROTOCOL_SHA,'protocol_pin')
    plan=json.loads(plan_path.read_text());release=json.loads(Path(release_path).read_text())
    need(release['status']=='GO' and release['plan']==pin(plan_path) and release['code_freeze']==plan['code_freeze'],'root_release')
    for record in plan['sources'].values():checked(record)
    author_freeze=read_json(plan['code_freeze']);need(author_freeze['code']==plan['code'],'author_freeze')
    for record in plan['code'].values():checked(record)
    contract=read_json(plan['sources']['input_contract'])
    need(contract['status']=='PASS_RETAINED_SEED42_OUTPUTS_REUSABLE_UNDER_THIS_EXACT_INPUT_CONTRACT','input_contract_status')
    need([c['id'] for c in contract['cases']]==IDS and contract['total_contexts']==6 and contract['total_captured_tiles']==150,'fixed_cases')
    need(contract['total_primary_positive']==86535 and contract['total_primary_negative']==434259,'fixed_class_totals')
    need(read_json(plan['sources']['input_verification'])['status']=='PASS','input_verification')
    need(read_json(plan['sources']['phase22_independent_verification'])['status']=='PASS','native_qualification')
    phase22=read_json(plan['sources']['phase22_plan']);helper=reference_helpers()
    schema=read_json(phase22['sources']['reference_schema'])
    expected_hashes=helper.qualified_hashes(phase22['stage1']['artifact'],schema)
    controller=json.loads((run/'controller-result.json').read_text())
    need(controller['status']=='PASS' and controller['stop'] is None and controller['returncode']==0 and controller['no_retry'] is True,'controller_status')
    need(controller['plan']==pin(plan_path) and controller['elapsed_seconds']<=240 and controller['external_peak_rss_bytes']<=4*1024**3,'controller_binding_limits')
    result=read_json(controller['result'])
    need(result['status']=='PASS' and result['plan']==pin(plan_path) and result['input_contract']==plan['sources']['input_contract'],'worker_bindings')
    need(result['model_constructions']==result['strict_loads']==1 and result['full_model_calls']==150
        and result['training_steps']==result['gradients']==0,'worker_call_counts')
    need(result['source_hashes_unchanged'] is True and result['state_unchanged'] is True
        and all(v==0 for v in result['forbidden_calls'].values()),'worker_invariants')
    need(result['elapsed_seconds']<=180 and result['cpu_seconds']<=360 and result['max_rss_bytes']<=4*1024**3,'worker_resource_limits')
    initial=read_json(result['load_verification']);final=read_json(result['final_state_verification'])
    need(initial['strict'] is True and initial['missing_keys']==[] and initial['unexpected_keys']==[], 'strict_native_load')
    need(initial['checkpoint']==phase22['stage1']['artifact'],'loaded_checkpoint')
    helper.verify_state(initial,expected_hashes,schema);helper.verify_state(final,expected_hashes,schema)
    need(set(result['runs'])==set(IDS),'worker_case_ids')
    published=json.loads(Path(metrics_path).read_text());independent={};arithmetic={};counts={};total_inputs=0
    need(published['status']=='PASS' and published['plan']==pin(plan_path) and published['native_result']==controller['result']
        and published['input_contract']==plan['sources']['input_contract'],'analysis_binding')
    need(published['model_calls']==published['training_steps']==published['threshold_fits']==published['threshold_classifications']==0
        and published['automatic_promotion'] is False and published['performance_pass_gate'] is None,'analysis_scope')
    need(published['elapsed_seconds']<=120 and published['max_rss_bytes']<=4*1024**3,'analysis_limits')
    for case in contract['cases']:
        cid=case['id'];meta=result['runs'][cid]
        saved=json.loads((run/'native'/cid/'metadata.json').read_text());need(saved==meta,'metadata_receipt_binding')
        need(meta['status']=='PASS' and meta['id']==cid and meta['role']=='reused_development_diagnostic','metadata_case')
        need(meta['input_contract']==plan['sources']['input_contract'] and meta['surface']==case['arrays']['surface']
            and meta['primary_mask']==case['arrays']['primary_mask'],'metadata_input_binding')
        need(meta['input_fingerprints_exact_before_forward'] is True and meta['surface_unchanged'] is True,'input_before_forward')
        source,baseline,fingerprints,baseline_check=load_context_sources(case,helper)
        need(meta['captured_tiles']==fingerprints,'native_fingerprints')
        inf=meta['inference'];need(inf['tile_count']==inf['scored_tile_count']==25,'case_forward_count')
        need([x['bbox_roi_xywh'] for x in inf['tiles']]==[x['bbox_roi_xywh'] for x in fingerprints],'case_actual_schedule')
        need(all(t['status']=='scored' for t in inf['tiles']),'unexpected_tile_skip')
        need(meta['primary_pixels']==meta['primary_evaluated_pixels']==case['counts']['primary'],'primary_not_dropped')
        native={key.replace('_','-'):load_array(record) for key,record in meta['arrays'].items()}
        need(set(native)=={'tile-logits','tile-probabilities','raw-scores','probability-sum','weight-sum','hann','evaluated-mask','uint8','decoded'},'native_array_set')
        arithmetic[cid]=dict(native=check_arrays(native,helper),seed42_blend=baseline_check)
        independent[cid]=context_metrics(case,source,baseline,native)
        counts[cid]={k:case['counts'][k] for k in ('primary','primary_positive','primary_negative')}
        total_inputs+=len(fingerprints)
        scalar_close(published['per_context'][cid],independent[cid],cid)
        pc=published['per_context'][cid]
        need(pc['counts']==counts[cid] and pc['primary_mask']==case['arrays']['primary_mask']
            and pc['labels']==case['arrays']['labels'] and pc['full_original_primary_support_preserved'] is True,'published_support')
        need(pc['score_inputs']==dict(native=meta['arrays']['raw_scores'],seed42=case['seed42']['scores']),'published_score_inputs')
        for role,raw in [('native',native['raw-scores']),('seed42',baseline['scores'])]:
            diagnostics=pc['diagnostic_arrays'][role]
            helper.exact(load_array(diagnostics['uint8']),helper.encode(raw),'diagnostic_uint8:'+role)
            helper.exact(load_array(diagnostics['decoded']),helper.decode(helper.encode(raw)),'diagnostic_decode:'+role)
    need(total_inputs==150 and sum(v['primary'] for v in counts.values())==520794,'support_and_input_totals')
    need(set(published['per_context'])==set(IDS),'published_cases')
    parents={p:parent_metrics(independent,p) for p in ('Paris1','Paris2')}
    for parent,value in parents.items():scalar_close(published['per_parent'][parent],value,'parent/'+parent)
    # Source, checkpoint and input identities remain pinned after all reads.
    for item in plan['sources'].values():checked(item)
    for case in contract['cases']:
        for record in case['arrays'].values():checked(record)
        for record in case['seed42'].values():
            if isinstance(record,dict) and 'path' in record:checked(record)
    closure=read_json(phase22['sources']['historical_closure']);checked(closure['authored_configuration'])
    vendor=read_json(phase22['sources']['current_vendor_manifest']);root=Path(phase22['sources']['current_vendor_manifest']['path']).parent
    for item in vendor['files']:need(pin(root/item['path'])['sha256']==item['sha256'],'vendor_final_pin')
    checked(phase22['sources']['baseline_adapter'])
    phase_bytes=sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file())
    need(phase_bytes<=256*1024**2,'phase_byte_limit')
    return dict(status='PASS',author_plan=pin(plan_path),root_release=pin(release_path),metrics=pin(metrics_path),
        result=controller['result'],per_context=independent,per_parent=parents,class_counts=counts,
        arithmetic=arithmetic,normalized_input_hashes_checked=total_inputs,state_hashes_checked=508,alias_pairs_checked=244,
        full_model_calls_verified=150,verifier_model_calls=0,network_calls=0,phase_bytes_before_receipt=phase_bytes,
        original_primary_pixels=520794,primary_positive=86535,primary_negative=434259,
        metric_tolerance=1e-12,scope='Six reused development contexts; implementation/metric verification, no performance pass gate.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--metrics',required=True);parser.add_argument('--root-release',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();signal.alarm(120);resource.setrlimit(resource.RLIMIT_CPU,(120,)*2);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
    def deny(event,args):
        if event.startswith('socket.'):raise Reject('network_forbidden')
    sys.addaudithook(deny);started=time.monotonic();cpu=time.process_time()
    try:result=verify(a.plan,a.run,a.metrics,a.root_release)
    except Exception as exc:
        result=dict(status='STOPPED',failure_type=type(exc).__name__,failure_code=str(exc) if isinstance(exc,(Reject,ValueError)) else 'unexpected_checker_exception')
    result.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    if result['max_rss_bytes']>4*1024**3:result.update(status='STOPPED',failure_code='checker_rss_limit')
    with Path(a.output).open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],receipt=pin(a.output))))
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
