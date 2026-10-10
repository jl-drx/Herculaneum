"""Post-completion-only tensor review; initializer and two new finals only.

One pinned CPU factory is constructed for schema/name inspection. No model
forward, backward, inference, strict model load or training is performed.
"""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

LAB=Path('/workspace/herculaneum-lab')
HELPER=LAB/'reports/eighth-experiment/metric-review/checkpoint_receipts.py'
HELPER_SHA='3556305cd2662b79ba87b1cf0a6f61161731e073dc5661ce0f8a14dfab2d705c'
OLD=LAB/'reports/ninth-experiment/review/result/independent-verification.json'
OLD_SHA='0cb4bf51046ea300d080bdd242526939c40604b34900f09960bb027798cd7a39'
NEW=('A_preserve','B_preserve')
HISTORICAL=('frozen_seed42','A_smooth','B_smooth')


def require(value,message):
    if not value:raise ValueError(message)


def verify_checkpoints(plan,workers):
    require(plan['kind']=='preservation-continuation-v1' and plan['ready_for_execution'] is True
            and plan['new_arms']==list(NEW),'Ready preservation plan required')
    require(all(workers[a]['status']=='PASS' for a in NEW),'Both real workers must be completed before tensor review')
    require(HELPER.stat().st_size==19165 and hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA,'Retained tensor helper differs')
    spec=importlib.util.spec_from_file_location('tenth_immutable_checkpoint_helper',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    receipts={}
    helper._pin(dict(path=str(HELPER),size_bytes=19165,sha256=HELPER_SHA),receipts)
    helper._pin(helper._record(Path(__file__)),receipts)
    def read(item):return json.loads(helper._pin(item,receipts).read_text())
    old=read(dict(path=str(OLD),size_bytes=190913,sha256=OLD_SHA))
    require(old['status']=='PASS' and old['checkpoint_review']['status']=='PASS','Ninth proof was not PASS')
    old_cp=old['checkpoint_review'];old_records={r['path']:r for r in old['verified_input_output_records_before_and_after']}
    require(plan['base_plan']['sha256']==old['plan_sha256'],'Preservation base differs from reviewed ninth plan')
    base=read(plan['base_plan'])
    for key in ('adaptation','development','diagnostic','initializer','teacher','schedule','exposure','helper','metrics','cutoff','calibration'):
        require(plan[key]==base[key],'Matched ninth source/mechanics differ: '+key)
    historical={}
    for arm in HISTORICAL:
        record=plan['historical_results'][arm];result=read(record)
        actual=helper._record(record['path'])
        require(old_records.get(actual['path'])==actual and workers[arm]==result,'Historical result differs from retained ninth proof')
        require(result['initial_state_sha256']==plan['historical_initial_state_sha256'],'Historical initializer differs')
        proof=dict(result_record=actual,unchanged_since_ninth_review=True,checkpoint_re_deserialized=False)
        if arm!='frozen_seed42':
            checkpoint=result['training']['checkpoint'];helper._pin(checkpoint,receipts)
            require(old_records.get(checkpoint['path'])==helper._record(checkpoint['path']),'Historical checkpoint byte identity differs')
            require(checkpoint==old_cp['adapted_checkpoints'][arm]['checkpoint'],'Historical checkpoint proof differs')
            proof['checkpoint_record']=checkpoint
        historical[arm]=proof
    for arm in NEW:
        worker=workers[arm]
        require(read(worker['plan'])==plan and worker['initializer']==plan['initializer'],'New worker input identity differs')
        require(worker['training']['steps']==32,'New worker update count differs')
    audited=read(dict(path=str(helper.AUDIT),size_bytes=helper.AUDIT_BYTES,sha256=helper.AUDIT_SHA))['models']['seed42']
    require(plan['initializer']['sha256']==audited['sha256'] and plan['initializer']['size_bytes']==audited['size_bytes'],'Audited initializer differs')
    import torch
    torch.set_num_threads(4)
    schema,groups,parameters,buffers,elements=helper._factory_schema(plan,audited,receipts,torch)
    schema_hash=hashlib.sha256(json.dumps(schema,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    names_hash=hashlib.sha256(json.dumps(sorted(parameters),separators=(',',':')).encode()).hexdigest()
    require(schema_hash==old_cp['factory_schema_sha256'] and names_hash==old_cp['canonical_parameter_names_sha256'],'Factory schema/name identities differ from ninth proof')
    payload,archive=helper._bounded_load(plan['initializer'],receipts,torch)
    state=helper._initializer_state(payload,audited)
    initial_storage=helper._check_state(state,schema,groups,audited,torch)
    initial_hash=helper._semantic_hash(state)
    initial_parameter_hash=helper._semantic_hash({name:state[name] for name in parameters})
    tensor_hashes={name:helper._tensor_hash(state[name]) for name in parameters}
    require(initial_hash==old_cp['initial_state_sha256']==plan['historical_initial_state_sha256']
            and initial_parameter_hash==old_cp['initial_parameter_sha256'],'Initializer tensor identity differs')
    del state,payload;gc.collect()
    finals={}
    for arm in NEW:
        training=workers[arm]['training']
        require(training['initial_state_sha256']==initial_hash and training['initial_parameter_sha256']==initial_parameter_hash,'Training initial tensor hashes differ')
        state,final_archive=helper._bounded_load(training['checkpoint'],receipts,torch)
        storage=helper._check_state(state,schema,groups,audited,torch)
        final_hash=helper._semantic_hash(state)
        parameter_hash=helper._semantic_hash({name:state[name] for name in parameters})
        require(final_hash==training['final_state_sha256'] and parameter_hash==training['final_parameter_sha256']
                and parameter_hash!=initial_parameter_hash,'Final tensor identity or parameter change differs')
        persisted=training['persisted_validation']
        require(persisted['weights_only'] is True and persisted['exact_tensor_values'] is True
                and persisted['missing_keys']==persisted['unexpected_keys']==[] and persisted['semantic_state_sha256']==final_hash,'Persisted strict-reload receipt differs')
        for key in ('tensor_count','unique_storage_count','alias_group_count','named_tensor_bytes','unique_storage_bytes'):
            require(storage[key]==persisted['storage_contract'][key],'Persisted storage count differs')
        changed=[name for name in parameters if helper._tensor_hash(state[name])!=tensor_hashes[name]]
        require(changed,'No independent parameter change measured')
        finals[arm]=dict(checkpoint=training['checkpoint'],archive=final_archive,storage_contract=storage,
                        semantic_state_sha256=final_hash,semantic_parameter_sha256=parameter_hash,
                        actual_parameters_changed=True,changed_parameter_names=changed,
                        changed_parameter_tensor_count=len(changed),elements_in_changed_tensors=sum(schema[n]['numel'] for n in changed))
        del state;gc.collect()
    for item in receipts.values():require(helper._record(item['path'])==item,'Tensor review input changed during verification')
    return dict(status='PASS',retained_ninth_proof_sha256=OLD_SHA,historical_reuse=historical,
                checkpoints_loaded=3,peak_concurrent_checkpoint_payloads=1,historical_final_checkpoints_deserialized=0,
                factory_constructions=1,factory_forward_calls=0,strict_model_load_calls=0,
                factory_schema_sha256=schema_hash,canonical_parameter_names_sha256=names_hash,
                parameter_element_count=elements,unique_parameter_tensor_count=len(parameters),unique_buffer_tensor_count=len(buffers),
                initial_state_sha256=initial_hash,initial_parameter_sha256=initial_parameter_hash,
                initializer_archive=archive,initializer_storage_contract=initial_storage,adapted_checkpoints=finals,
                verified_source_and_checkpoint_records_before_and_after=list(receipts.values()),
                forward_calls=0,backward_calls=0,inference_calls=0,network_requests=0)
