"""Synthetic independent audit of fixed-coefficient manual-to-retired KL allocation."""
import json
import math
import numpy as np
from review_objective import objective, probability, require


def allocation(student,teacher,target,manual,retired,arm):
    m,r=np.asarray(manual),np.asarray(retired)
    require(m.dtype==r.dtype==np.bool_ and m.shape==r.shape and m.any() and r.any() and not(m&r).any(),
            'Nonempty disjoint manual and explicitly retired-reference strata required')
    require(arm in ('A','B'),'Unknown allocation arm')
    if arm=='A':return objective(student,teacher,target,m,m|r,mode='manual',weight=2.)
    return objective(student,teacher,target,m,m|r,mode='stratified',weight=1.,unlabeled_weight=1.)


def fixtures():
    z=np.asarray([.8,-.4,1.2,-2.,.2,1.7,7.])
    t=np.asarray([-.7,.3,-.4,.6,1.1,-.2,-7.])
    y=np.asarray([1.,0.,-1.,-1.,-1.,-1.,float('nan')])
    m=np.asarray([1,1,0,0,0,0,0],dtype=bool)
    r=np.asarray([0,0,1,1,1,1,0],dtype=bool)
    base=objective(z,t,y,m,m|r,mode='manual',weight=0.)
    values={arm:allocation(z,t,y,m,r,arm) for arm in ('A','B')};checks=[]
    for arm,answer in values.items():
        for index in range(z.size):
            plus,minus=z.copy(),z.copy();plus[index]+=1e-5;minus[index]-=1e-5
            finite=(allocation(plus,t,y,m,r,arm)['full']-allocation(minus,t,y,m,r,arm)['full'])/2e-5
            require(abs(finite-answer['derivative'][index])<=2e-9,'Allocation derivative finite difference differs')
        checks.append(arm+': seven finite differences, including outside-union zero')
        require(answer['supervised']==base['supervised'],'Manual BCE changed across allocation arms')
        checks.append(arm+': unchanged manual BCE')
        changed=y.copy();changed[~m]=[0.,1.,999.,-17.,float('nan')]
        second=allocation(z,t,changed,m,r,arm)
        require(answer['full']==second['full'] and np.array_equal(answer['derivative'],second['derivative']),
                'Retired or unsupported targets influenced allocation')
        checks.append(arm+': all nonmanual targets ignored')
        zz,tt=z.copy(),t.copy();zz[~(m|r)]=-1000.;tt[~(m|r)]=1000.
        second=allocation(zz,tt,y,m,r,arm)
        require(answer['full']==second['full'] and np.array_equal(answer['derivative'],second['derivative']),
                'Outside-union output influenced allocation')
        checks.append(arm+': outside-union student/teacher mutation ignored')
    manual_A=values['A']['derivative'][m]-base['derivative'][m]
    manual_B=values['B']['derivative'][m]-base['derivative'][m]
    require(np.all(np.abs(manual_A-2*manual_B)<=1e-12),'Manual teacher protection not halved')
    require(np.all(np.abs(values['B']['derivative'][r]-(probability(z[r])-probability(t[r]))/r.sum())<=1e-12)
            and np.all(values['A']['derivative'][r]==0), 'Retired-stratum derivative differs')
    checks+=['manual KL derivative exactly halved','new retired derivative uses its own stratum mean']
    uniform=objective(z,t,y,m,m|r,mode='uniform_valid',weight=2.)
    require(abs(uniform['full']-values['B']['full'])>.001,'Unequal-stratum weighting conflated with uniform union')
    checks.append('unequal-size group means differ from twice uniform union mean')
    same_z=np.full(7,.7);same_t=np.full(7,-.4)
    common={arm:allocation(same_z,same_t,y,m,r,arm) for arm in ('A','B')}
    require(abs(common['A']['full']-common['B']['full'])<=1e-12,
            'Constant per-output KL does not conserve total group coefficient')
    checks.append('constant per-output KL matches total group coefficients two')
    for label,bad_m,bad_r in [('empty retired',m,np.zeros_like(r)),('empty manual',np.zeros_like(m),r),
                              ('overlap',m,m|r)]:
        for arm in ('A','B'):
            try:allocation(z,t,y,bad_m,bad_r,arm)
            except ValueError:checks.append(arm+': reject '+label)
            else:raise ValueError('Malformed proposed allocation accepted')
    return dict(status='PASS',synthetic_only=True,check_count=len(checks),checks=checks,
                hypotheses=dict(A='BCE_M + 2 KL_M',B='BCE_M + KL_M + KL_retired'),
                shared_manual_BCE=True,group_coefficient_sum_A=2,group_coefficient_sum_B=2,
                manual_KL_derivative_ratio_B_over_A=.5,retired_target_values_used=False,
                empty_retired_policy='STOP',geometry_validity_claim=False,scientific_design_selected_by_reviewer=False,
                scientific_array_reads=0,model_calls=0,optimizer_steps=0,network_calls=0)


if __name__=='__main__':print(json.dumps(fixtures(),indent=2,sort_keys=True,allow_nan=False))
