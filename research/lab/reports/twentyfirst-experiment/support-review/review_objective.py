"""Independent NumPy-only support-objective mathematics; synthetic inputs only."""
import json
import math
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(logit):
    return np.exp(-np.logaddexp(0., -np.asarray(logit, dtype=np.float64)))


def fields(student, teacher, target, manual, valid):
    z, t, y = [np.asarray(x, dtype=np.float64) for x in (student, teacher, target)]
    m, g = np.asarray(manual), np.asarray(valid)
    require(z.shape == t.shape == y.shape == m.shape == g.shape and z.ndim == 1,
            'Aligned one-dimensional synthetic outputs required')
    require(m.dtype == g.dtype == np.bool_ and m.any() and g.any() and not (m & ~g).any(),
            'Nonempty manual subset of supplied support required')
    require(np.isfinite(z).all() and np.isfinite(t).all() and np.isin(y[m], [0, 1]).all(),
            'Finite logits and binary manual targets required')
    lp, ln = -np.logaddexp(0., -z), -np.logaddexp(0., z)
    lq, lqn = -np.logaddexp(0., -t), -np.logaddexp(0., t)
    p, q, qn = np.exp(lp), np.exp(lq), np.exp(lqn)
    bce = -(.25 + .5 * y[m]) * lp[m] - (.75 - .5 * y[m]) * ln[m]
    divergence = q * (lq-lp) + qn * (lqn-ln)
    return z, t, y, m, g, p, q, bce, divergence


def objective(student, teacher, target, manual, valid, *, mode, weight=1., unlabeled_weight=.5):
    require(mode in ('manual', 'uniform_valid', 'stratified') and
            math.isfinite(weight) and weight >= 0 and math.isfinite(unlabeled_weight) and unlabeled_weight >= 0,
            'Declared synthetic objective and nonnegative finite weights required')
    z,t,y,m,g,p,q,bce,kl = fields(student,teacher,target,manual,valid)
    derivative = np.zeros_like(z)
    derivative[m] = (p[m]-(.25+.5*y[m])) / int(m.sum())
    supervised = math.fsum(float(x) for x in bce) / bce.size
    if mode == 'manual':
        preservation = math.fsum(float(x) for x in kl[m]) / int(m.sum())
        derivative[m] += weight*(p[m]-q[m])/int(m.sum())
    elif mode == 'uniform_valid':
        preservation = math.fsum(float(x) for x in kl[g]) / int(g.sum())
        derivative[g] += weight*(p[g]-q[g])/int(g.sum())
    else:
        u = g & ~m
        preservation = math.fsum(float(x) for x in kl[m])/int(m.sum())
        derivative[m] += weight*(p[m]-q[m])/int(m.sum())
        # Frozen illustrative policy: an empty U contributes zero, with no renormalization.
        if u.any():
            preservation += unlabeled_weight*math.fsum(float(x) for x in kl[u])/int(u.sum())
            derivative[u] += weight*unlabeled_weight*(p[u]-q[u])/int(u.sum())
    require((derivative[~g] == 0).all(), 'Unsupported outputs contributed derivatives')
    return dict(supervised=supervised, preservation=preservation, full=supervised+weight*preservation,
                derivative=derivative)


def fixtures():
    checks=[]
    z=np.asarray([.7,-1.1,.3,1.8,-3.2,9.])
    t=np.asarray([-.3,.2,-.7,.6,1.2,-9.])
    y=np.asarray([1.,0.,-1.,999.,-12.,-1.])
    m=np.asarray([True,True,False,False,False,False])
    g=np.asarray([True,True,True,True,True,False])
    for mode in ('manual','uniform_valid','stratified'):
        value=objective(z,t,y,m,g,mode=mode)
        for index in range(len(z)):
            plus,minus=z.copy(),z.copy();plus[index]+=1e-5;minus[index]-=1e-5
            finite=(objective(plus,t,y,m,g,mode=mode)['full']-
                    objective(minus,t,y,m,g,mode=mode)['full'])/2e-5
            require(abs(finite-value['derivative'][index]) <= 2e-9, 'Finite-difference derivative differs')
        checks.append(mode+': six finite-difference derivatives, including unsupported output')
        changed=y.copy();changed[~m]=[-99,0,1,float('nan')]
        other=objective(z,t,changed,m,g,mode=mode)
        require(other['full'] == value['full'] and np.array_equal(other['derivative'],value['derivative']),
                'Nonmanual targets influenced objective')
        checks.append(mode+': nonmanual target changes ignored')
        changed_z=z.copy();changed_z[~g]=-999.
        changed_t=t.copy();changed_t[~g]=999.
        other=objective(changed_z,changed_t,y,m,g,mode=mode)
        require(other['full'] == value['full'] and np.array_equal(other['derivative'],value['derivative']),
                'Unsupported logits influenced objective')
        checks.append(mode+': unsupported student/teacher changes ignored')
        equal=objective(t,t,y,m,g,mode=mode)
        require(equal['preservation']==0 and equal['full']==equal['supervised'], 'Teacher identity KL differs')
        checks.append(mode+': identical teacher gives zero KL')
        doubled=objective(np.tile(z,2),np.tile(t,2),np.tile(y,2),np.tile(m,2),np.tile(g,2),mode=mode)
        require(abs(doubled['full']-value['full'])<=1e-12 and
                np.all(np.abs(doubled['derivative']-np.tile(value['derivative']/2,2))<=1e-12),
                'Repeated-sample mean scaling differs')
        checks.append(mode+': duplicating all samples preserves mean and halves each derivative')
    manual=objective(z,t,y,m,g,mode='manual');uniform=objective(z,t,y,m,g,mode='uniform_valid')
    supervised=objective(z,t,y,m,g,mode='manual',weight=0)
    require(np.all(np.abs((uniform['derivative'][m]-supervised['derivative'][m])-
                         (manual['derivative'][m]-supervised['derivative'][m])*m.sum()/g.sum())<=1e-12),
            'Manual preservation dilution ratio differs')
    require(np.all(manual['derivative'][g&~m]==0) and np.all(uniform['derivative'][g&~m]!=0),
            'Supplied unlabeled support did not acquire the expected teacher gradient')
    checks += ['exact |M|/|G| manual KL dilution','manual-only zero and expanded nonzero supplied-unlabeled gradients']
    for mode in ('uniform_valid','stratified'):
        same=objective(z,t,y,m,m.copy(),mode=mode)
        require(same['full']==manual['full'] and np.array_equal(same['derivative'],manual['derivative']),
                'Empty U did not recover manual objective')
        checks.append(mode+': empty U recovers manual objective without division by zero')
    z2=np.asarray([math.log(3.)]);t2=np.asarray([math.log(9.)]);mask=np.ones(1,dtype=bool)
    result=objective(z2,t2,np.ones(1),mask,mask,mode='uniform_valid')
    forward=.9*math.log(.9/.75)+.1*math.log(.1/.25)
    reverse=.75*math.log(.75/.9)+.25*math.log(.25/.1)
    require(abs(result['preservation']-forward)<=1e-12 and abs(forward-reverse)>.01,
            'KL teacher-to-student direction differs')
    checks.append('asymmetric closed-form Bernoulli KL direction')
    extreme=objective(np.asarray([-1000.,1000.]),np.asarray([1000.,-1000.]),
                      np.asarray([0.,1.]),np.ones(2,dtype=bool),np.ones(2,dtype=bool),mode='uniform_valid')
    require(all(math.isfinite(extreme[key]) for key in ('full','supervised','preservation')) and
            np.isfinite(extreme['derivative']).all(), 'Extreme-logit stability differs')
    checks.append('stable extreme-logit KL, BCE and derivatives')
    # Equal batch weighting is deliberately different from pooling unequal supports.
    small=[2.,4.];large=[1.]*6
    equal_batch=(sum(small)/len(small)+sum(large)/len(large))/2
    pooled=(sum(small)+sum(large))/(len(small)+len(large))
    require(equal_batch==2. and pooled==1.5 and equal_batch!=pooled, 'Batch and pixel reductions conflated')
    checks.append('unequal support: equal-batch 2.0 versus pooled-pixel 1.5')
    for label,mask_m,mask_g in [('empty manual',np.zeros_like(m),g),('empty supplied support',m,np.zeros_like(g)),
                               ('manual outside supplied support',m,np.asarray([False,True,True,True,True,False]))]:
        try:objective(z,t,y,mask_m,mask_g,mode='uniform_valid')
        except ValueError:checks.append('reject '+label)
        else:raise ValueError('Malformed support accepted')
    return dict(status='PASS',synthetic_only=True,check_count=len(checks),checks=checks,
                manual_preservation_derivative_ratio=float(m.sum()/g.sum()),
                example_weights=dict(lambda_value=1.,stratified_unlabeled_weight=.5,scientific_choice=False),
                scientific_array_reads=0,torch_imports=0,model_calls=0,optimizer_steps=0,network_calls=0)


if __name__=='__main__':
    print(json.dumps(fixtures(),indent=2,sort_keys=True,allow_nan=False))
