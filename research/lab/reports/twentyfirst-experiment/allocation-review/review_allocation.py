"""Independent manual/retired Bernoulli objectives; no producer numerical imports."""
import math
import numpy as np
from review_math import require, support_losses, average

ARMS=('manual_double','manual_retired')
FIELDS={'base','kl_manual','kl_retired','objective_A','objective_B','full','hard_bce',
        'probability_drift_manual','probability_drift_retired'}


def support_contract(student,teacher,manual,target,retired,arm):
    require(arm in ARMS and student.shape==teacher.shape==manual.shape==target.shape==retired.shape,
            'Aligned allocation arrays and declared arm required')
    require(manual.dtype==retired.dtype==np.bool_ and manual.any() and retired.any() and not(manual&retired).any(),
            'Nonempty disjoint manual and retired strata required')
    require(np.isfinite(student).all() and np.isfinite(teacher).all() and np.isin(target[manual],[0,1]).all(),
            'Finite logits and binary participating manual labels required')


def allocation_record(student,teacher,manual,target,retired,arm):
    support_contract(student,teacher,manual,target,retired,arm)
    terms=support_losses(student[manual],teacher[manual],target[manual])
    z,t=student[retired].astype(np.float64),teacher[retired].astype(np.float64)
    lp,ln=-np.logaddexp(0.,-z),-np.logaddexp(0.,z)
    lq,lqn=-np.logaddexp(0.,-t),-np.logaddexp(0.,t)
    q,qn=np.exp(lq),np.exp(lqn)
    kr=average(q*(lq-lp)+qn*(lqn-ln))
    drift=average(np.abs(np.exp(lp)-q))
    base,km=terms['smoothed_bce'],terms['kl']
    a,b=base+2*km,base+km+kr
    return dict(base=base,kl_manual=km,kl_retired=kr,objective_A=a,objective_B=b,
                full=a if arm=='manual_double' else b,hard_bce=terms['hard_bce'],
                probability_drift_manual=terms['probability_drift'],probability_drift_retired=drift)


def allocation_derivative(student,teacher,manual,target,retired,arm):
    support_contract(student,teacher,manual,target,retired,arm)
    p=np.exp(-np.logaddexp(0.,-student.astype(np.float64)))
    q=np.exp(-np.logaddexp(0.,-teacher.astype(np.float64)))
    out=np.zeros(student.shape,dtype=np.float64)
    coefficient=2 if arm=='manual_double' else 1
    out[manual]=(p[manual]-(.25+.5*target[manual])+coefficient*(p[manual]-q[manual]))/int(manual.sum())
    if arm=='manual_retired':out[retired]=(p[retired]-q[retired])/int(retired.sum())
    return out


def diagnostic_difference(first,second):
    require(first.dtype==second.dtype==np.float32 and first.shape==second.shape and
            np.isfinite(first).all() and np.isfinite(second).all(),'Aligned finite FP32 diagnostic arrays required')
    changed=int(np.count_nonzero(first.view(np.uint32)!=second.view(np.uint32)))
    return dict(shape=list(first.shape),scalar_count=int(first.size),bitwise_equal=changed==0,
                bitwise_changed_scalars=changed,maximum_absolute_difference=float(np.max(np.abs(first.astype(np.float64)-second.astype(np.float64)),initial=0.)),
                scope='Diagnostic only; no inter-arm numerical acceptance threshold')


def fixture_checks():
    z=np.asarray([.8,-.4,1.2,-2.,.2,1.7,7.],dtype=np.float64)
    t=np.asarray([-.7,.3,-.4,.6,1.1,-.2,-7.],dtype=np.float64)
    y=np.asarray([1.,0.,-1.,float('nan'),-1.,-1.,-1.])
    m=np.asarray([1,1,0,0,0,0,0],dtype=bool);r=np.asarray([0,0,1,1,1,1,0],dtype=bool)
    values={};checks=[]
    for arm in ARMS:
        value=allocation_record(z,t,m,y,r,arm);gradient=allocation_derivative(z,t,m,y,r,arm)
        require(set(value)==FIELDS,'Objective schema differs')
        for index in range(z.size):
            plus,minus=z.copy(),z.copy();plus[index]+=1e-5;minus[index]-=1e-5
            numerical=(allocation_record(plus,t,m,y,r,arm)['full']-allocation_record(minus,t,m,y,r,arm)['full'])/2e-5
            require(abs(numerical-gradient[index])<=2e-9,'Allocation finite difference differs')
        checks.append(arm+': seven finite-difference derivatives')
        outside=~m if arm=='manual_double' else ~(m|r)
        require((gradient[outside]==0).all(),'Unsupported output derivative differs')
        checks.append(arm+': correct exact-zero derivative support')
        changed=y.copy();changed[~m]=[0.,1.,-999.,float('nan'),999.]
        require(allocation_record(z,t,m,changed,r,arm)==value and
                np.array_equal(allocation_derivative(z,t,m,changed,r,arm),gradient),'Nonmanual targets influenced objective')
        checks.append(arm+': retired/unknown labels ignored')
        identity=allocation_record(t,t,m,y,r,arm)
        require(identity['kl_manual']==identity['kl_retired']==identity['probability_drift_manual']==identity['probability_drift_retired']==0.,
                'Teacher identity differs')
        checks.append(arm+': exact teacher identity KL and drift')
        common=allocation_record(z,t,m,y,r,'manual_retired' if arm=='manual_double' else 'manual_double')
        require(all(value[key]==common[key] for key in FIELDS-{'full'}),'Common diagnostics differ by arm')
        require(value['full']==value['objective_A' if arm=='manual_double' else 'objective_B'],'Own/common objective linkage differs')
        checks.append(arm+': shared diagnostics and correct own objective')
        values[arm]=(value,gradient)
    p=np.exp(-np.logaddexp(0.,-z));supervised=(p[m]-(.25+.5*y[m]))/m.sum()
    require(np.all(np.abs(values['manual_double'][1][m]-supervised-2*(values['manual_retired'][1][m]-supervised))<=1e-12),
            'Manual KL coefficient not halved')
    checks.append('manual KL derivative is halved at fixed total group coefficient')
    km,kr=values['manual_retired'][0]['kl_manual'],values['manual_retired'][0]['kl_retired']
    require(abs((km+kr)-2*(m.sum()*km+r.sum()*kr)/(m.sum()+r.sum()))>.001,'Separate groups conflated with union mean')
    checks.append('unequal-size group weighting differs from uniform union')
    for bad_m,bad_r in [(np.zeros_like(m),r),(m,np.zeros_like(r)),(m,m|r)]:
        for arm in ARMS:
            try:allocation_record(z,t,bad_m,y,bad_r,arm)
            except ValueError:checks.append(arm+': reject empty/overlapping role')
            else:raise ValueError('Malformed role support accepted')
    first=np.asarray([0.,1.],dtype=np.float32);second=np.asarray([-0.,np.nextafter(np.float32(1),np.float32(2))],dtype=np.float32)
    diagnostic=diagnostic_difference(first,second)
    require(diagnostic['bitwise_changed_scalars']==2 and diagnostic['maximum_absolute_difference']==float(np.spacing(np.float32(1))),
            'Diagnostic-only inter-arm difference lost signed zero or ULP change')
    checks.append('first-step differences diagnostic only, including signed zero and one ULP')
    return dict(status='PASS',synthetic_only=True,checks=checks,check_count=len(checks),scientific_array_reads=0,
                model_calls=0,checkpoint_decodes=0,training_replays=0)
