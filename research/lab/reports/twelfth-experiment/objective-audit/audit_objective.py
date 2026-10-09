"""Frozen-plan NumPy reconstruction of saved preupdate objectives; no model I/O."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import socket
import sys
import time

def offline(*args,**kwargs): raise RuntimeError('Network disabled in objective audit')
socket.create_connection=offline; socket.socket.connect=offline; socket.socket.connect_ex=offline
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='4'
import numpy as np

ARMS=['A_preserve','B_preserve']
WINDOWS={'1-8':list(range(1,9)),'9-16':list(range(9,17)),'17-24':list(range(17,25)),
         '25-32':list(range(25,33)),'all-1-32':list(range(1,33))}

def require(condition,message):
    if not condition: raise ValueError(message)

def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)

def verify(item):
    require(record(item['path'])==item,'Input identity changed: '+Path(item['path']).name)
    return Path(item['path'])

def support_terms(z,t,y):
    """Pure supported-output algebra; epsilon=.5, teacher||student KL, no clamp."""
    z=np.asarray(z,dtype=np.float64); t=np.asarray(t,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    require(z.shape==t.shape==y.shape and z.size>0,'Nonempty aligned support required')
    require(np.isfinite(z).all() and np.isfinite(t).all() and np.all((y==0)|(y==1)),'Finite logits/hard targets required')
    s=.25+.5*y; q=np.exp(-np.logaddexp(0.,-t)); spz=np.logaddexp(0.,z); spt=np.logaddexp(0.,t)
    bce=spz-s*z; teacher_bce=spt-s*t
    divergence=spz-spt-q*(z-t); linear=(q-s)*(z-t)
    optimum_probability=(s+q)/2.; optimum_logit=np.log(optimum_probability)-np.log1p(-optimum_probability)
    spo=np.logaddexp(0.,optimum_logit)
    optimum_full=spo-s*optimum_logit+spo-spt-q*(optimum_logit-t)
    avg=lambda a:float(np.mean(a,dtype=np.float64))
    base_delta=avg(bce)-avg(teacher_bce); full_delta=avg(bce)+avg(divergence)-avg(teacher_bce)
    result=dict(count=int(z.size),positive=int((y==1).sum()),negative=int((y==0).sum()),
        student_smoothed_BCE=avg(bce),initializer_smoothed_BCE=avg(teacher_bce),
        student_hard_BCE=avg(spz-y*z),initializer_hard_BCE=avg(spt-y*t),
        KL=avg(divergence),linear_drift=avg(linear),base_delta=base_delta,full_delta=full_delta,
        student_full=avg(bce)+avg(divergence),initializer_full=avg(teacher_bce),
        base_decomposition_residual=base_delta-avg(divergence)-avg(linear),
        full_decomposition_residual=full_delta-2*avg(divergence)-avg(linear),
        free_logit_optimum_full=avg(optimum_full),
        student_gap_to_free_optimum=avg(bce)+avg(divergence)-avg(optimum_full),
        initializer_gap_to_free_optimum=avg(teacher_bce)-avg(optimum_full),
        optimum_probability_min=float(optimum_probability.min()),optimum_probability_max=float(optimum_probability.max()))
    return result

def augment(array,code):
    require(type(code) is int and 0<=code<8,'D4 code differs')
    value=np.rot90(array,code%4,axes=(-2,-1))
    return np.ascontiguousarray(np.flip(value,axis=-1) if code>=4 else value)

def load_regions(plan):
    regions=[]
    for case in plan['regions']:
        data={k:np.load(verify(v),mmap_mode='r',allow_pickle=False) for k,v in case['arrays'].items()}
        require(all(a.shape==(384,384) for a in data.values()),'Regional shapes differ')
        for name in ['inklabels','supervision_mask','validation_mask']:
            require(data[name].dtype==np.uint8,'Raw annotation dtype differs')
        for name in ['supervision_mask','validation_mask']:
            require(set(np.unique(data[name]).tolist())<={0,1},'Mask encoding differs')
        require(data['observed'].dtype==np.bool_ and data['teacher43_evaluated'].dtype==np.bool_
            and data['teacher43_scores'].dtype==np.float32,'Observed/teacher-map dtype differs')
        m=data['observed']&(data['supervision_mask']>0)&~(data['validation_mask']>0)
        scores=data['teacher43_scores']; finite=np.isfinite(scores)
        require(np.all((scores[finite]>=0)&(scores[finite]<=1)),'Teacher43 scores outside range')
        p=data['observed']&data['teacher43_evaluated']&finite&~m
        manual=np.full(m.shape,-1.,dtype=np.float32); pseudo=np.full(p.shape,-1.,dtype=np.float32)
        manual[m]=data['inklabels'][m]; pseudo[p]=(scores[p]>=.5).astype(np.float32)
        require(np.all((manual[m]==0)|(manual[m]==1)) and not np.any(m&p),'Hard target/support contract differs')
        regions.append(dict(id=case['id'],m=m,p=p,y=manual,h=pseudo))
    return regions

def masks_for_step(regions,step):
    masks=[[],[]]; targets=[[],[]]; exposures=[]
    require([patch['region'] for patch in step['patches']]==[0,1],'Fixed batch ordering differs')
    for patch in step['patches']:
        r=patch['region']; x,y,w,h=patch['bbox']; code=patch['augmentation']
        require(w==h==128 and 0<=x<=256 and 0<=y<=256 and x%64==y%64==0,'Scheduled tile bounds differ')
        exposure=dict(region=r)
        for index,(mask_name,target_name,role) in enumerate([('m','y','manual'),('p','h','pseudo')]):
            mask=augment(regions[r][mask_name][y:y+h,x:x+w],code)
            target=augment(regions[r][target_name][y:y+h,x:x+w],code)
            masks[index].append(mask); targets[index].append(target)
            exposure[role]=dict(positive=int((mask&(target==1)).sum()),negative=int((mask&(target==0)).sum()))
        require(min(exposure['manual'].values())>=256,'Original manual class gate differs')
        exposures.append(exposure)
    return [np.stack(v) for v in masks],[np.stack(v) for v in targets],exposures

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True); args=parser.parse_args()
    started=time.perf_counter(); plan_record=record(args.plan); plan=json.loads(verify(plan_record).read_text())
    require(plan['kind']=='retained-online-objective-audit-v1' and plan['ready_for_execution'] is True,'Audit not frozen ready')
    require(plan['windows']==WINDOWS and plan['arms']==ARMS,'Predeclared scope differs')
    require(plan['tolerances']==dict(float32_scalar_absolute=2e-6,float64_identity_absolute=1e-12,
        KL_mean_floor=-1e-6,first_logit_absolute=1e-7),'Audit tolerances differ')
    require(plan['limits']==dict(wall_seconds=60,cpu_seconds=120,rss_bytes=1024**3,
        address_space_bytes=8*1024**3,output_bytes=8*1024**2),'Audit resource limits differ')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__,threads=4),
        'Frozen audit runtime differs')
    require('torch' not in sys.modules,'Tensor/model runtime must not be imported')
    require(plan['code']==record(Path(__file__).resolve()),'Audit source changed')
    for item in plan['verified_inputs']: verify(item)
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2); resource.setrlimit(resource.RLIMIT_CPU,(120,)*2)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Objective audit wall cap'))); signal.alarm(60)
    regions=load_regions(plan); schedule=json.loads(verify(plan['schedule']).read_text())
    require([s['step'] for s in schedule]==list(range(1,33)),'Schedule differs')
    args.output_dir.mkdir(parents=True,exist_ok=False); arms={}; csv_rows=[]; teacher_reference=None; maximum_errors={}
    for arm in ARMS:
        inputs=plan['arm_inputs'][arm]; coefficient=0. if arm=='A_preserve' else .25
        z=np.load(verify(inputs['student_logits']),mmap_mode='r',allow_pickle=False)
        t=np.load(verify(inputs['teacher_logits']),mmap_mode='r',allow_pickle=False)
        require(all(a.dtype==np.float32 and a.shape==(32,2,1,128,128) and np.isfinite(a).all() for a in [z,t]),'Saved logit contract differs')
        if teacher_reference is None: teacher_reference=t
        else: require(np.array_equal(t,teacher_reference),'Initializer teacher logits differ between arms')
        require(float(np.max(np.abs(z[0]-t[0])))<=1e-7,'First-step student/initializer differs')
        logs=[json.loads(line) for line in verify(inputs['steps']).read_text().splitlines()]
        old=[json.loads(line) for line in verify(inputs['ninth_steps']).read_text().splitlines()]
        require([r['step'] for r in logs]==[r['step'] for r in old]==list(range(1,33)),'Step receipt scope differs')
        rows=[]
        for index,(step,logged,historical) in enumerate(zip(schedule,logs,old)):
            require(logged['patches']==historical['patches']==step['patches'],'Matched schedule/augmentations differ')
            (m,p),(y,h),exposures=masks_for_step(regions,step)
            require(exposures==logged['exposures']==historical['exposures'],'Reconstructed support exposure differs')
            zz=z[index,:,0].astype(np.float64); tt=t[index,:,0].astype(np.float64)
            manual=support_terms(zz[m],tt[m],y[m]); pseudo=support_terms(zz[p],tt[p],h[p]); strata=[]; manual_groups=[]
            for r in [0,1]:
                for name,mask,target in [('manual',m,y),('pseudo',p,h)]:
                    for label in [0,1]:
                        select=mask[r]&(target[r]==label); count=int(select.sum())
                        term=support_terms(zz[r][select],tt[r][select],target[r][select]) if count else None
                        strata.append(dict(region=r,support=name,hard_class=label,count=count,
                            fraction_of_own_support=count/int(mask.sum()),terms=term))
                        if name=='manual': manual_groups.append(term['student_smoothed_BCE'])
            g=lambda key:manual[key]+coefficient*pseudo[key]
            values=dict(U_student=manual['student_smoothed_BCE'],U_initializer=manual['initializer_smoothed_BCE'],
                P_student=pseudo['student_smoothed_BCE'],P_initializer=pseudo['initializer_smoothed_BCE'],
                hard_base_student=g('student_hard_BCE'),hard_base_initializer=g('initializer_hard_BCE'),
                base_student=g('student_smoothed_BCE'),base_initializer=g('initializer_smoothed_BCE'),base_delta=g('base_delta'),
                full_student=g('student_full'),full_initializer=g('initializer_full'),full_delta=g('full_delta'),
                weighted_KL=g('KL'),weighted_linear_drift=g('linear_drift'),free_optimum_full=g('free_logit_optimum_full'),
                student_gap_to_free_optimum=g('student_gap_to_free_optimum'),initializer_gap_to_free_optimum=g('initializer_gap_to_free_optimum'),
                base_decomposition_residual=g('base_decomposition_residual'),full_decomposition_residual=g('full_decomposition_residual'),
                ninth_base_receipt=historical['total_loss'],ninth_U_receipt=historical['loss_U'],ninth_P_receipt=historical['loss_P'])
            expected=dict(loss_U=values['U_student'],loss_P=values['P_student'],loss_R=float(np.mean(manual_groups)),
                teacher_U=values['U_initializer'],teacher_P=values['P_initializer'],loss_D_M=manual['KL'],loss_D_P=pseudo['KL'],
                original_objective=values['base_student'],total_loss=values['full_student'])
            errors={key:abs(value-logged[key]) for key,value in expected.items()}
            for key,error in errors.items():
                tolerance=1e-12 if key in ['loss_D_M','loss_D_P'] else 2e-6
                require(error<=tolerance,'Saved scalar reconstruction differs: '+arm+'/'+str(index+1)+'/'+key)
                maximum_errors[key]=max(maximum_errors.get(key,0.),error)
            require(abs(historical['total_loss']-(historical['loss_U']+coefficient*historical['loss_P']))<=2e-6,'Ninth receipt coefficient differs')
            for term in [manual,pseudo]+[s['terms'] for s in strata if s['terms'] is not None]:
                require(term['KL']>=-1e-6 and abs(term['base_decomposition_residual'])<=1e-12
                    and abs(term['full_decomposition_residual'])<=1e-12,'KL/decomposition identity failed')
                require(min(term['student_gap_to_free_optimum'],term['initializer_gap_to_free_optimum'])>=-1e-12,'Free-logit optimum identity failed')
            require(logged['unknown_output_gradient_max']==0. and np.isfinite(logged['preclip_global_gradient_norm'])
                and logged['preclip_global_gradient_norm']>0.,'Historical gradient receipt differs')
            values['ninth_base_delta_vs_same_input_initializer_receipt']=values['ninth_base_receipt']-values['base_initializer']
            values['base_delta_vs_ninth_receipt']=values['base_student']-values['ninth_base_receipt']
            row=dict(step=index+1,patches=step['patches'],exposures=exposures,manual=manual,pseudo=pseudo,strata=strata,
                values=values,reconstruction_absolute_errors=errors,unknown_active_output_count=int((~m if coefficient==0 else ~(m|p)).sum()))
            rows.append(row); csv_rows.append(dict(arm=arm,step=index+1,**values))
        windows={}
        for name,steps in WINDOWS.items():
            chosen=[rows[i-1] for i in steps]
            windows[name]=dict(steps=steps,equal_step_means={key:float(np.mean([r['values'][key] for r in chosen])) for key in values},
                support_counts={support:{label:sum(r[support][label] for r in chosen) for label in ['count','positive','negative']} for support in ['manual','pseudo']},
                steps_with_positive_base_delta=sum(r['values']['base_delta']>0 for r in chosen),
                steps_with_positive_full_delta=sum(r['values']['full_delta']>0 for r in chosen))
        arms[arm]=dict(pseudo_coefficient=coefficient,steps=rows,windows=windows,ninth_evidence='Scalar receipts only; all-step ninth logits were not retained')
    for item in plan['verified_inputs']: verify(item)
    require(record(args.plan)==plan_record,'Audit plan changed during reconstruction')
    output=dict(status='PASS',plan=plan_record,arms=arms,maximum_logged_scalar_reconstruction_errors=maximum_errors,
        teacher_logits_bitwise_identical_between_arms=True,
        original_objective_is_same_smoothed_base_as_optimized=True,original_objective_excludes_preservation=True,
        hard_BCE_secondary_only=True,free_optimum_scope='Independent output-space mathematical diagnostic, not an attainable shared-network or generalization claim',
        online_scope='Preupdate evolving-model snapshots, not final-model loss or immediate before/after parameter-step evidence',
        checkpoint_loads=0,model_calls=0,backward_calls=0,optimizer_calls=0,network_calls=0,
        wall_seconds=time.perf_counter()-started,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
    require(output['peak_rss_bytes']<=1024**3,'Objective audit RSS acceptance exceeded')
    (args.output_dir/'audit.json').write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (args.output_dir/'steps.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    require(sum(f.stat().st_size for f in args.output_dir.iterdir() if f.is_file())<=8*1024**2,'Audit output exceeds8MiB')
    signal.alarm(0); print(json.dumps(dict(status='PASS',audit=record(args.output_dir/'audit.json'))))

if __name__=='__main__': main()
