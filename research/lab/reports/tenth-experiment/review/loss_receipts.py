"""Reconstruct every completed tenth loss from saved logits; pure NumPy only."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

LAB=Path('/workspace/herculaneum-lab')
NORMALIZER=LAB/'vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py'
NEW=('A_preserve','B_preserve')


def require(value,message):
    if not value: raise ValueError(message)


def numpy_state_hash(values):
    digest=hashlib.sha256()
    for key,array in sorted(values.items()):
        array=np.ascontiguousarray(array)
        dtype={np.dtype('float32'):'torch.float32',np.dtype('bool'):'torch.bool'}[array.dtype]
        digest.update(json.dumps([key,list(array.shape),dtype],separators=(',',':')).encode()+b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def sigmoid(x):
    e=np.exp(-np.abs(x));return np.where(x>=0,1/(1+e),e/(1+e))


def softplus(x):
    return np.maximum(x,0)+np.log1p(np.exp(-np.abs(x)))


def bce(z,mask,target):
    yy=.25+.5*target[mask].astype(np.float64)
    return float(np.mean(softplus(z[mask])-yy*z[mask],dtype=np.float64))


def review_losses(plan,workers,pin):
    """pin(path,record=None) verifies identity and returns a Path; no writes."""
    def load(item):return np.load(pin(item['path'],item),allow_pickle=False)
    def read(item):return json.loads(pin(item['path'],item).read_text())
    spec=importlib.util.spec_from_file_location('tenth_review_normalizer',pin(NORMALIZER))
    normalizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(normalizer)
    source=[]
    for case in plan['adaptation']:
        a={key:load(item) for key,item in case['inputs'].items()}
        require(case['physical_group']=='PHerc0139','Unexpected adaptation specimen')
        shape=a['surface'].shape[1:]
        require(a['surface'].dtype==np.uint8 and a['surface'].shape[0]==21,'Source array differs')
        require(a['acquisition_observed_mask'].dtype==bool and a['acquisition_observed_mask'].all(),'Observed source support differs')
        for key in ('inklabels','supervision_mask','validation_mask'):
            require(a[key].shape==shape and set(np.unique(a[key]))<={0,1},'Binary source contract differs')
        m=(a['supervision_mask']>0)&(a['validation_mask']==0)&a['acquisition_observed_mask']
        scores,evaluated=load(case['teacher_scores']),load(case['teacher_evaluated'])
        require(scores.dtype==np.float32 and evaluated.dtype==bool and scores.shape==evaluated.shape==shape,'Hard teacher shape/dtype differs')
        p=a['acquisition_observed_mask']&evaluated&np.isfinite(scores)&~m
        y=np.full(shape,-1.,np.float32);y[m]=a['inklabels'][m]
        py=np.full(shape,-1.,np.float32);py[p]=(scores[p]>=.5).astype(np.float32)
        source.append((a,m,y,p,py))
    require(len(source)==2,'Exactly two source regions required')
    schedule=read(plan['schedule']);require(len(schedule)==32,'Exactly32 scheduled steps required')
    def aug(a,code):
        value=np.rot90(a,code%4,axes=(-2,-1))
        return np.ascontiguousarray(np.flip(value,axis=-1) if code>=4 else value)
    batches=[]
    for index,step in enumerate(schedule,1):
        require(step['step']==index and [p['region'] for p in step['patches']]==[0,1],'Schedule order differs')
        batch={key:[] for key in ('input','manual_mask','manual_hard_target','pseudo_mask','pseudo_hard_target')};exposures=[]
        for patch in step['patches']:
            region=patch['region'];a,m,y,p,py=source[region];x,yy,w,h=patch['bbox'];code=patch['augmentation']
            require(patch['bbox'] in plan['exposure'][region]['eligible_tiles'] and 0<=code<8 and w==h==128,'Frozen tile/D4 differs')
            sl=np.s_[yy:yy+h,x:x+w]
            image=normalizer.normalize_robust(a['surface'][2:19,yy:yy+h,x:x+w])
            batch['input'].append(aug(np.ascontiguousarray(image,dtype=np.float32),code)[None])
            for key,value in (('manual_mask',m),('manual_hard_target',y),('pseudo_mask',p),('pseudo_hard_target',py)):
                batch[key].append(aug(value[sl],code)[None])
            exposures.append(dict(region=region,manual=dict(positive=int((m[sl]&(y[sl]==1)).sum()),negative=int((m[sl]&(y[sl]==0)).sum())),
                                  pseudo=dict(positive=int((p[sl]&(py[sl]==1)).sum()),negative=int((p[sl]&(py[sl]==0)).sum()))))
        arrays={key:np.stack(value) for key,value in batch.items()}
        require(np.isfinite(arrays['input']).all(),'Normalized source nonfinite')
        fingerprint=numpy_state_hash(arrays)
        del arrays['input']
        batches.append((arrays,fingerprint,exposures))
    summaries={};teacher_arrays={};max_errors={};reconstructed={}
    for arm in NEW:
        worker=workers[arm];training=worker['training'];old_arm=arm.replace('_preserve','_smooth')
        old_training=workers[old_arm]['training']
        require(training['smoothing_epsilon']==.5 and training['preservation_lambda']==1.
                and training['schedule']==plan['schedule'] and training['exposure']==plan['exposure'],'Training objective/schedule contract differs')
        rows=[json.loads(line) for line in pin(training['steps_log']['path'],training['steps_log']).read_text().splitlines()]
        old_rows=[json.loads(line) for line in pin(old_training['steps_log']['path'],old_training['steps_log']).read_text().splitlines()]
        require(len(rows)==len(old_rows)==training['steps']==32,'Training step count differs')
        student=load(training['preupdate_logits']['student']);teacher=load(training['preupdate_logits']['teacher'])
        require(student.shape==teacher.shape==(32,2,1,128,128) and student.dtype==teacher.dtype==np.float32,'Preupdate logits shape/dtype differs')
        require(np.isfinite(student).all() and np.isfinite(teacher).all(),'Retained logits nonfinite')
        teacher_arrays[arm]=teacher
        computed=[]
        for index,(row,old,step,batch) in enumerate(zip(rows,old_rows,schedule,batches)):
            arrays,fingerprint,exposures=batch
            require(row['step']==old['step']==step['step'] and row['patches']==old['patches']==step['patches'],'Matched row schedule differs')
            require(row['exposures']==old['exposures']==exposures and row['input_target_sha256']==fingerprint,'Independent batch fingerprint/exposure differs')
            require(row['smoothing_epsilon']==.5 and row['preservation_lambda']==1.
                    and row['teacher_rng_preserved'] is True and row['teacher_has_gradients'] is False,'Perstep objective/teacher guard differs')
            m,y,p,py=[arrays[key] for key in ('manual_mask','manual_hard_target','pseudo_mask','pseudo_hard_target')]
            z,t=student[index].astype(np.float64),teacher[index].astype(np.float64)
            q,prob=sigmoid(t),sigmoid(z)
            # Different algebra from training logaddexp/Bregman expression.
            kl=q*(softplus(-z)-softplus(-t))+(1-q)*(softplus(z)-softplus(t))
            u,ps=bce(z,m,y),bce(z,p,py)
            group=[]
            for r in range(2):
                for label in (0.,1.):
                    mask=m[r]&(y[r]==label);require(mask.any(),'Missing region/manual class')
                    group.append(bce(z[r],mask,y[r]))
            dm,dp=float(kl[m].mean()),float(kl[p].mean())
            original=u if arm=='A_preserve' else u+.25*ps
            values=dict(loss_U=u,loss_P=ps,loss_R=math.fsum(group)/4,loss_D_M=dm,loss_D_P=dp,
                        total_loss=original+dm+(0 if arm=='A_preserve' else .25*dp),
                        teacher_U=bce(t,m,y),teacher_P=bce(t,p,py),original_objective=original,
                        probability_drift_manual=float(np.abs(prob[m]-q[m]).mean()),
                        probability_drift_pseudo=float(np.abs(prob[p]-q[p]).mean()))
            require(min(dm,dp)>=-1e-6,'Reconstructed mean KL below roundoff guard')
            for key,value in values.items():
                require(math.isfinite(value) and math.isfinite(row[key]),'Nonfinite reconstructed/saved loss')
                error=abs(value-row[key]);max_errors[key]=max(max_errors.get(key,0.),error)
                tolerance=1e-12 if key in ('loss_D_M','loss_D_P','probability_drift_manual','probability_drift_pseudo') else 2e-6
                require(error<=tolerance,'Raw logit reconstruction differs: '+arm+'/'+str(index+1)+'/'+key)
            require(row['unknown_output_gradient_max']==0 and math.isfinite(row['preclip_global_gradient_norm']) and row['preclip_global_gradient_norm']>0,'Gradient receipt differs')
            probe=row['parameter_update_probe'];require(math.isfinite(probe['before']) and math.isfinite(probe['after']) and probe['before']!=probe['after'],'Update probe differs')
            if index==0:
                require(np.max(np.abs(z-t))<=1e-7 and np.max(np.abs(kl))<=1e-7,'Firststep teacher/student mismatch')
                grad=np.zeros_like(z);grad[m]=(prob[m]-q[m])/int(m.sum())
                if arm=='B_preserve':grad[p]=.25*(prob[p]-q[p])/int(p.sum())
                require(np.max(np.abs(grad))<=1e-7,'Firststep preservation derivative differs')
                for key in ('loss_U','loss_P'):
                    require(abs(row[key]-old[key])<=1e-7,'Historical firststep smoothed loss differs')
                require(abs(row['preclip_global_gradient_norm']-old['preclip_global_gradient_norm'])<=1e-6,'Historical firststep gradient norm differs')
                sentinel=read(training['first_step_sentinel']);prior=read(plan['historical_first_sentinels'][old_arm])
                require(sentinel['initial_state_sha256']==plan['historical_initial_state_sha256']
                        and sentinel['schedule']==plan['schedule'] and sentinel['patches']==step['patches']
                        and sentinel['input_target_sha256']==prior['input_target_sha256']==fingerprint,'Independent firststep sentinel binding differs')
                for mask,target,key in ((m,y,'hard_U'),(p,py,'hard_P')):
                    hard=float(np.mean(softplus(z[mask])-target[mask].astype(np.float64)*z[mask]))
                    require(abs(hard-sentinel[key])<=2e-6 and abs(sentinel[key]-prior[key])<=1e-7,'Independent hard first loss differs')
                for key,actual in (('teacher_student_logit_maxdiff',float(np.max(np.abs(z-t)))),
                                   ('preservation_element_maxabs',float(np.max(np.abs(kl)))),
                                   ('preservation_output_gradient_maxabs',float(np.max(np.abs(grad))))):
                    require(abs(sentinel[key]-actual)<=1e-12,'Reconstructed firststep sentinel differs')
            computed.append(dict(step=index+1,**values))
        teacher_guard=training['teacher']
        require(teacher_guard['initial_state_sha256']==teacher_guard['final_state_sha256']==plan['historical_initial_state_sha256'],'Teacher state receipt differs')
        require(all(teacher_guard[k] is True for k in ('no_gradients','disjoint_storage','rng_preserved')),'Teacher isolation guard differs')
        def average(key):return math.fsum(row[key] for row in computed[-8:])/8
        current_u=average('loss_U');current_original=average('original_objective')
        historical_u=math.fsum(r['loss_U'] for r in old_rows[-8:])/8
        historical_original=math.fsum(r['total_loss'] for r in old_rows[-8:])/8
        teacher_u=average('teacher_U');teacher_original=teacher_u+(0 if arm=='A_preserve' else .25*average('teacher_P'))
        summaries[arm]=dict(steps=32,matched_historical_arm=old_arm,allstep_source_fingerprints_reconstructed=True,
            last_eight=dict(steps=list(range(25,33)),current_U=current_u,current_original_objective=current_original,
                historical_U=historical_u,historical_original_objective=historical_original,
                frozen_teacher_U=teacher_u,frozen_teacher_original_objective=teacher_original,
                U_change_vs_historical=current_u-historical_u,original_change_vs_historical=current_original-historical_original,
                U_change_vs_initializer=current_u-teacher_u,original_change_vs_initializer=current_original-teacher_original,
                D_M=average('loss_D_M'),D_P=average('loss_D_P'),probability_drift_manual=average('probability_drift_manual'),
                probability_drift_pseudo=average('probability_drift_pseudo')),
            first_step_input_target_sha256=batches[0][1],teacher_guards=teacher_guard)
        reconstructed[arm]=computed
    require(np.array_equal(teacher_arrays[NEW[0]],teacher_arrays[NEW[1]]),'Frozen teacher predictions differ between matched arms')
    return dict(status='PASS',all_64_losses_reconstructed_from_raw_logits=True,
                cross_arm_teacher_logits_exactly_equal=True,maximum_absolute_reconstruction_errors=max_errors,
                arms=summaries,reconstructed_steps=reconstructed,
                benchmark_loss_code_imported=False,model_forward_calls=0,backward_calls=0,network_requests=0,
                scope='Matched preupdate online diagnostics; not final-model accuracy or independent transfer')
