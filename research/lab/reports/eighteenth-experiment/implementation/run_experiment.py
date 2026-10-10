"""Offline bounded phase18: matched adapter training, then final development inference."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode=True


def offline(*args,**kwargs):
    raise RuntimeError('Network forbidden in phase18 experiment')


socket.create_connection=offline
socket.socket.connect=offline
socket.socket.connect_ex=offline

import numpy as np
import torch
from depth_common import (require, record, verify, read_json, write_json, state_hash,
                          isolation_state, equal_batch_mean, rss, output_bytes, load_plan)
from depth_adapter import ARMS, ResidualAdapter, compose, manual_terms
from depth_data import runtime, training_batches, development_case


def setup(plan, phase):
    limits=plan['limits']
    signal.alarm(limits[phase+'_wall_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(limits['worker_cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['output_bytes'],)*2)
    require(platform.python_version()==plan['runtime']['python'] and np.__version__==plan['runtime']['numpy']
            and torch.__version__==plan['runtime']['torch'],'Pinned runtime differs')
    torch.set_num_threads(limits['threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    seed=plan['training']['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)


def measure(adapter,optimizer,batches,indices):
    before=isolation_state(adapter,optimizer);rows=[];logits=[];residuals=[]
    with torch.no_grad():
        for i in indices:
            batch=batches[i]
            z,delta=compose(adapter,batch['representation'],batch['teacher'])
            _,terms=manual_terms(z,batch['teacher'],batch['mask'],batch['target'])
            logits.append(z.numpy().copy());residuals.append(delta.numpy().copy());rows.append(terms)
    after=isolation_state(adapter,optimizer)
    require(before==after,'Measurement mutated state/optimizer/gradients/mode/RNG')
    return np.stack(logits),np.stack(residuals),rows,dict(before=before,after=after,no_grad=True,unchanged=True)


def strict_reload(adapter,checkpoint,expected_hash):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    expected=adapter.state_dict()
    require(isinstance(saved,dict) and set(saved)==set(expected),'Adapter checkpoint keys differ')
    storage=[]
    for name,value in saved.items():
        require(isinstance(value,torch.Tensor) and value.layout==torch.strided
                and value.dtype==torch.float32 and value.device.type=='cpu'
                and value.shape==expected[name].shape and value.is_contiguous()
                and value.storage_offset()==0 and value.untyped_storage().nbytes()==value.numel()*value.element_size()
                and bool(torch.isfinite(value).all()),'Checkpoint tensor/storage differs: '+name)
        storage.append(dict(name=name,shape=list(value.shape),dtype=str(value.dtype),bytes=value.untyped_storage().nbytes()))
    require(state_hash(saved)==expected_hash,'Saved adapter semantic hash differs')
    strict=adapter.load_state_dict(saved,strict=True)
    require(not strict.missing_keys and not strict.unexpected_keys and state_hash(adapter.state_dict())==expected_hash,
            'Strict adapter reload differs')
    return dict(weights_only=True,strict=True,finite=True,compact_storages=storage,
                parameter_count=sum(v.numel() for v in saved.values()),state_sha256=expected_hash)


def training_worker(plan,path,arm,directory):
    started=time.perf_counter();setup(plan,'training');directory.mkdir(parents=True,exist_ok=False)
    context=runtime(plan);batches=training_batches(plan,arm,context)
    torch.manual_seed(plan['training']['seed'])
    adapter=ResidualAdapter().train();initial=state_hash(adapter.state_dict())
    initial_rng=isolation_state(adapter)['torch_rng']
    train=plan['training']
    require(train==dict(steps=32,batch_size=2,seed=20261008,learning_rate=.001,optimizer='AdamW',
                       betas=[.9,.999],eps=1e-8,weight_decay=0,gradient_clip_norm=1.,
                       loss_smoothing=.5,kl_coefficient=1.,foreach=False,fused=False,
                       fixed_updates=[1,8,16,32]),'Training configuration differs')
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=train['learning_rate'],betas=tuple(train['betas']),
                                eps=train['eps'],weight_decay=train['weight_decay'],foreach=False,fused=False)
    named=list(adapter.named_parameters());parameters=[p for _,p in named]
    require(list(optimizer.param_groups[0]['params'])==parameters and len(optimizer.param_groups)==1
            and not list(adapter.named_buffers()),'Optimizer membership/buffer contract differs')
    parameter_order=[dict(name=name,shape=list(p.shape),numel=p.numel()) for name,p in named]
    def vector():return np.concatenate([p.detach().numpy().ravel() for p in parameters]).copy()
    def moment(key):
        return np.concatenate([optimizer.state[p][key].numpy().ravel() if p in optimizer.state
                               else np.zeros(p.numel(),dtype=np.float32) for p in parameters]).copy()
    traces={name:np.empty((32,801),dtype=np.float32) for name in
            ['parameters_before','parameters_after','gradients_before_clip','gradients_after_clip',
             'exp_avg_before','exp_avg_after','exp_avg_sq_before','exp_avg_sq_after']}
    require(bool((adapter.head.weight==0).all()) and bool((adapter.head.bias==0).all())
            and all(b['representation'].shape==(2,3,17,128,128) and bool(torch.isfinite(b['representation']).all())
                    and torch.equal((b['teacher']+torch.zeros_like(b['teacher'])).view(torch.int32),b['teacher'].view(torch.int32))
                    for b in batches),
            'Algebraic all32-batch zero projection proof failed')
    baseline_rows=[manual_terms(b['teacher'],b['teacher'],b['mask'],b['target'])[1] for b in batches]
    baseline=dict(per_batch=baseline_rows,equal_batch_means=equal_batch_mean(baseline_rows),
                  scope='verified cached seed42 logits; no new backbone forwards')
    arrays={}
    for kind in ['logits','residuals']:
        for stage,shape in [('preupdate',(32,2,1,128,128)),('postupdate',(32,2,1,128,128)),
                            ('fixed-risk',(4,32,2,1,128,128))]:
            name=stage+'-'+kind
            arrays[name]=np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=np.float32,shape=shape)
    arrays['logit-gradients']=np.lib.format.open_memmap(directory/'logit-gradients.npy',mode='w+',dtype=np.float32,shape=(32,2,1,128,128))
    evaluations={};sentinel=None;logpath=directory/'steps.jsonl'
    with logpath.open('x') as stream:
        for i,batch in enumerate(batches):
            clock=time.perf_counter()
            require(hashlib.sha256(batch['image'].numpy().tobytes()).hexdigest()==batch['input_sha256']
                    and hashlib.sha256(batch['representation'].numpy().tobytes()).hexdigest()==batch['feature_sha256'],
                    'Training input/features mutated')
            optimizer.zero_grad(set_to_none=True)
            z,delta=compose(adapter,batch['representation'],batch['teacher'])
            arrays['preupdate-logits'][i]=z.detach().numpy();arrays['preupdate-residuals'][i]=delta.detach().numpy()
            z.retain_grad();total,pre=manual_terms(z,batch['teacher'],batch['mask'],batch['target'])
            if i==0:
                require(torch.equal(z.detach().view(torch.int32),batch['teacher'].view(torch.int32))
                        and bool((delta.detach()==0).all()),'Zero-head initialization is not exact cached baseline')
            traces['parameters_before'][i]=vector()
            traces['exp_avg_before'][i]=moment('exp_avg');traces['exp_avg_sq_before'][i]=moment('exp_avg_sq')
            total.backward()
            arrays['logit-gradients'][i]=z.grad.detach().numpy()
            require(z.grad is not None and bool((z.grad[~batch['mask']]==0).all()),'Nonmanual output contributed gradient')
            require(all(b['teacher'].grad is None and not b['teacher'].requires_grad for b in batches),
                    'Frozen backbone cache acquired gradient')
            stem_grad=sum(float(p.grad.detach().double().square().sum()) for p in adapter.stem.parameters())**.5
            head_grad=sum(float(p.grad.detach().double().square().sum()) for p in adapter.head.parameters())**.5
            if i==0:require(stem_grad==0 and head_grad>0,'Initial zero head gradient path differs')
            if i==1:require(stem_grad>0,'Hidden residual stem did not receive second-step gradients')
            traces['gradients_before_clip'][i]=np.concatenate([p.grad.detach().numpy().ravel() for p in parameters])
            norm=torch.nn.utils.clip_grad_norm_(parameters,train['gradient_clip_norm'],error_if_nonfinite=True,foreach=False)
            traces['gradients_after_clip'][i]=np.concatenate([p.grad.detach().numpy().ravel() for p in parameters])
            clipping_coefficient=float(torch.clamp(train['gradient_clip_norm']/(norm+1e-6),max=1.))
            require(float(norm)>0,'Zero adapter gradient')
            previous={name:p.detach().clone() for name,p in adapter.named_parameters()}
            optimizer.step()
            traces['parameters_after'][i]=vector()
            traces['exp_avg_after'][i]=moment('exp_avg');traces['exp_avg_sq_after'][i]=moment('exp_avg_sq')
            require(all(float(optimizer.state[p]['step'])==i+1 for p in parameters),'AdamW step counter differs')
            update=sum(float((p.detach().double()-previous[name].double()).square().sum()) for name,p in adapter.named_parameters())**.5
            require(update>0 and all(bool(torch.isfinite(p).all()) for p in adapter.parameters()),'Invalid adapter update')
            post_z,post_r,post_rows,guard=measure(adapter,optimizer,batches,[i])
            arrays['postupdate-logits'][i]=post_z[0];arrays['postupdate-residuals'][i]=post_r[0]
            if i==0:
                sentinel=dict(initial_adapter_state_sha256=initial,initial_rng_sha256=initial_rng,
                              baseline_logits_bitwise_equal=True,initial_residual_all_zero=True,
                              initial_stem_gradient_l2=stem_grad,initial_head_gradient_l2=head_grad,
                              preclip_gradient_norm=float(norm),input_target_sha256=batch['input_target_sha256'])
                write_json(directory/'first-step-sentinel.json',sentinel)
            if i+1 in train['fixed_updates']:
                fixed_z,fixed_r,rows,isolation=measure(adapter,optimizer,batches,list(range(32)))
                require(np.array_equal(fixed_z[i].view(np.uint32),post_z[0].view(np.uint32))
                        and np.array_equal(fixed_r[i].view(np.uint32),post_r[0].view(np.uint32)),
                        'Immediate post/fixed-risk predictions differ')
                idx=train['fixed_updates'].index(i+1)
                arrays['fixed-risk-logits'][idx]=fixed_z;arrays['fixed-risk-residuals'][idx]=fixed_r
                evaluations[str(i+1)]=dict(per_batch=rows,equal_batch_means=equal_batch_mean(rows),isolation=isolation,
                                           current_batch_post_bitwise_equal=True)
            row={k:batch[k] for k in ['step','patches','exposures','input_target_sha256','input_sha256','feature_sha256','manual_sha256']}
            row.update(pre=pre,post=post_rows[0],preclip_global_gradient_norm=float(norm),
                       stem_gradient_l2=stem_grad,head_gradient_l2=head_grad,parameter_update_l2=update,
                       clipping_coefficient=clipping_coefficient,
                       unknown_output_gradient_max=0.,measurement_isolation=guard,
                       wall_seconds=time.perf_counter()-clock,peak_rss_bytes=rss())
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
            require(rss()<=plan['limits']['rss_bytes'] and output_bytes(directory)<=plan['limits']['output_bytes'],'Training resource cap exceeded')
            del z,delta,total,previous
    for array in arrays.values():array.flush()
    arrays.clear();final=state_hash(adapter.state_dict())
    np.savez(directory/'optimizer-traces.npz',**traces)
    write_json(directory/'optimizer-contract.json',dict(parameter_order=parameter_order,
               parameter_count=801,optimizer_class='torch.optim.AdamW',learning_rate=train['learning_rate'],
               betas=train['betas'],eps=train['eps'],weight_decay=0,foreach=False,fused=False,
               optimizer_membership_exact=True,cached_backbone_members=0,
               clip_foreach=False,clip_epsilon=1e-6,step_counters=list(range(1,33))))
    require(final!=initial,'Final adapter state unchanged')
    checkpoint=directory/'final-adapter.pth';torch.save(adapter.state_dict(),checkpoint)
    adapter.zero_grad(set_to_none=True)
    reload_proof=strict_reload(adapter,checkpoint,final)
    require(checkpoint.stat().st_size<=16384,'Adapter checkpoint size differs')
    load_plan(path,execute=True)
    result=dict(status='PASS',phase='training',arm=arm,plan=record(path),
                initial_adapter_state_sha256=initial,final_adapter_state_sha256=final,
                initial_rng_sha256=initial_rng,checkpoint=record(checkpoint),
                checkpoint_reload=reload_proof,steps=record(logpath),first_step_sentinel=record(directory/'first-step-sentinel.json'),
                optimizer_traces=record(directory/'optimizer-traces.npz'),optimizer_contract=record(directory/'optimizer-contract.json'),
                arrays={name:record(directory/(name+'.npy')) for name in ['logit-gradients']+[s+'-'+k for k in ['logits','residuals'] for s in ['preupdate','postupdate','fixed-risk']]},
                baseline=baseline,fixed_evaluations=evaluations,teacher_caches=plan['teacher_caches'],
                backbone=dict(state_sha256=plan['backbone_state_sha256'],immutable_cached_evaluation=True,
                              loads=0,forwards=0,backwards=0,updates=0,gradients_absent=True),
                actual_calls=dict(adapter_forwards=192,backwards=32,optimizer_steps=32),
                initialization_proof=dict(all32_feature_tensors_finite=True,head_weight_and_bias_exact_zero=True,
                                          all32_residuals_algebraically_zero=True,all32_cached_add_zero_bitwise_identity=True,
                                          first_scheduled_forward_bitwise_baseline=True,
                                          additional_adapter_initialization_forwards=0),
                pseudo_data_use='PHerc0139 historical identity verification only; no pseudo loss or teacher term',
                paris_training_reads=0,wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss())
    write_json(directory/'result.json',result);signal.alarm(0)


def development_worker(plan,path,arm,directory,training_result,training_verification):
    started=time.perf_counter();setup(plan,'development');directory.mkdir(parents=True,exist_ok=False)
    verification=read_json(record(training_verification))
    require(verification['status']=='PASS'
            and verification['training_results']==record(directory.parent.parent/'training-results.json'),
            'Independent saved-output training verification is required before development')
    training_controller=read_json(verification['training_results'])
    require(training_controller['status']=='PASS' and training_controller['plan']==record(path)
            and training_controller['results']['training'][arm]==record(training_result),
            'Development arm was not covered by independent training verification')
    training=read_json(record(training_result))
    require(training['status']=='PASS' and training['phase']=='training' and training['arm']==arm
            and training['plan']==record(path),'Training result lineage differs')
    adapter=ResidualAdapter().eval()
    reload_proof=strict_reload(adapter,verify(training['checkpoint']),training['final_adapter_state_sha256'])
    before=isolation_state(adapter);context=runtime(plan);runs={}
    for case in plan['cases']:
        target=directory/case['id'];target.mkdir()
        scores,evaluated,metadata,capture=development_case(plan,case,context,adapter,arm)
        for name,array in [('scores',scores),('evaluated-mask',evaluated),
                           ('tile-logits',np.stack(capture.logits)),('tile-residuals',np.stack(capture.residuals))]:
            np.save(target/(name+'.npy'),array,allow_pickle=False)
        item=dict(scores=record(target/'scores.npy'),evaluated=record(target/'evaluated-mask.npy'),
                  logits=record(target/'tile-logits.npy'),residuals=record(target/'tile-residuals.npy'),
                  inference=metadata,baseline_metadata=case['baseline_metadata'],
                  captured_tiles=[dict(bbox_roi_xywh=row['bbox_roi_xywh'],input_sha256=h,feature_sha256=f)
                                  for row,h,f in zip(metadata['tiles'],capture.inputs,capture.feature_hashes)],
                  fixed_primary_mask=case['arrays']['primary_mask'],input_identity_exact=True,
                  role='reused development diagnostic',nominal_depth_coordinate=True)
        write_json(target/'metadata.json',item);runs[case['id']]=item
        require(rss()<=plan['limits']['rss_bytes'] and output_bytes(directory)<=plan['limits']['output_bytes'],'Development resource cap exceeded')
        del scores,evaluated,capture;gc.collect()
    after=isolation_state(adapter);require(before==after,'Development changed state/gradients/mode/RNG')
    load_plan(path,execute=True)
    result=dict(status='PASS',phase='development',arm=arm,plan=record(path),training_result=record(training_result),
                checkpoint_reload=reload_proof,runs=runs,isolation=dict(before=before,after=after,unchanged=True),
                actual_calls=dict(adapter_forwards=150,backbone_forwards=0,backwards=0,optimizer_steps=0,threshold_fits=0),
                wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss())
    write_json(directory/'result.json',result);signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--worker-phase',choices=['training','development']);parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--training-result',type=Path)
    parser.add_argument('--stage',choices=['training','development'])
    parser.add_argument('--training-verification',type=Path)
    args=parser.parse_args();plan,path=load_plan(args.plan,execute=True)
    if args.worker_phase:
        require(args.arm is not None,'Worker arm missing')
        if args.worker_phase=='training':training_worker(plan,path,args.arm,args.output_dir)
        else:
            require(args.training_result is not None and args.training_verification is not None,
                    'Frozen final training result/independent verification missing')
            development_worker(plan,path,args.arm,args.output_dir,args.training_result,args.training_verification)
        return
    require(args.stage is not None,'Choose the training or development controller stage')
    started=time.perf_counter();results={};workers=[]
    if args.stage=='training':
        args.output_dir.mkdir(parents=True,exist_ok=False)
    else:
        require(args.output_dir.is_dir() and args.training_verification is not None,
                'Development requires completed training and independent saved-output verification')
        training_results=record(args.output_dir/'training-results.json')
        verified=read_json(record(args.training_verification))
        require(verified['status']=='PASS' and verified['training_results']==training_results,
                'Independent training verification does not bind this exact completed run')
        completed=read_json(training_results)
        require(completed['status']=='PASS' and completed['plan']==record(path),'Training controller result differs')
        results['training']=completed['results']['training']
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for phase in [args.stage]:
        results[phase]={}
        for arm in ARMS:
            target=args.output_dir/phase/arm;target.parent.mkdir(exist_ok=True)
            remaining=plan['limits'][phase+'_controller_wall_seconds']-(time.perf_counter()-started)
            require(remaining>0,'Total experiment wall cap exhausted')
            command=[sys.executable,str(Path(__file__).resolve()),'--plan',str(path),'--output-dir',str(target),'--worker-phase',phase,'--arm',arm]
            if phase=='development':
                command+=['--training-result',results['training'][arm]['path'],
                          '--training-verification',str(args.training_verification.resolve())]
            command_record=dict(phase=phase,arm=arm,command=command,started_at_unix=time.time(),
                                controller_elapsed_seconds=time.perf_counter()-started)
            command_path=args.output_dir/(phase+'-'+arm+'-command.json')
            write_json(command_path,command_record)
            with (args.output_dir/(phase+'-'+arm+'.log')).open('xb') as stream:
                try:
                    completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,
                                             timeout=min(remaining,plan['limits'][phase+'_wall_seconds']+5))
                except subprocess.TimeoutExpired as error:
                    write_json(args.output_dir/(phase+'-'+arm+'-failure.json'),
                               dict(status='FAIL',reason='controller timeout',command=record(command_path),
                                    timeout_seconds=error.timeout,retries=0))
                    raise
            if completed.returncode!=0:
                write_json(args.output_dir/(phase+'-'+arm+'-failure.json'),
                           dict(status='FAIL',reason='worker nonzero exit',command=record(command_path),
                                exit_code=completed.returncode,retries=0))
            require(completed.returncode==0,'Worker stopped; zero retries or substitutions authorized')
            pin=record(target/'result.json');receipt=read_json(pin)
            require(receipt['status']=='PASS' and receipt['phase']==phase and receipt['arm']==arm
                    and receipt['plan']==record(path),'Worker result lineage differs')
            results[phase][arm]=pin;workers.append(dict(phase=phase,arm=arm,result=pin,exit_code=0,command=record(command_path)))
            require(output_bytes(Path(__file__).resolve().parent.parent)<=plan['limits']['output_bytes'],'Aggregate phase output byte cap exceeded')
            print(json.dumps(dict(status='PASS',phase=phase,arm=arm,result=pin)),flush=True)
        if phase=='training':
            first,second=[read_json(results[phase][arm]) for arm in ARMS]
            require(first['initial_adapter_state_sha256']==second['initial_adapter_state_sha256']
                    and first['initial_rng_sha256']==second['initial_rng_sha256'], 'Matched initialization/RNG differs')
            write_json(args.output_dir/'training-complete.json',dict(status='PASS',plan=record(path),training=results[phase],
                        matched_initial_adapter_and_rng=True,development_predictions=0))
    load_plan(path,execute=True)
    require(time.perf_counter()-started<=plan['limits'][args.stage+'_controller_wall_seconds'],'Controller wall cap exceeded')
    summary=dict(status='PASS',phase=args.stage,plan=record(path),results=results,workers=workers,
               training_complete=record(args.output_dir/'training-complete.json'),retries=0,
               all_required_arms_retained=True,wall_seconds=time.perf_counter()-started,
               actual_calls=dict(adapter_forwards=384 if args.stage=='training' else 300,
                                 backwards=64 if args.stage=='training' else 0,
                                 optimizer_steps=64 if args.stage=='training' else 0,
                                 backbone_forwards=0,threshold_fits=0))
    if args.stage=='development':
        summary['training_verification']=record(args.training_verification)
        summary['training_results']=record(args.output_dir/'training-results.json')
        summary['all_six_contexts_both_final_arms_retained']=True
    write_json(args.output_dir/(args.stage+'-results.json'),summary)


if __name__=='__main__':main()
