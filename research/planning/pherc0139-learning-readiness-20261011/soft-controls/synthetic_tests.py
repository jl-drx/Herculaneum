#!/usr/bin/env python3
"""Prospective fixtures for the finite operator study; no real target reads."""
from datetime import datetime,timezone
from fractions import Fraction
import json
import numpy as np
import analysis as a


def close(actual,expected,tol=1e-11):
    np.testing.assert_allclose(actual,expected,atol=tol,rtol=0)


def test_constant_and_origin():
    old=a.dependency()
    crop=np.full((32,40),103,np.uint8)
    origin=(100,200)
    uv=np.array([[[26.3,51.1],[27.8,53.2]],[[28.1,54.8],[29.2,55.4]]])
    for operator in a.OPERATORS:
        value,valid=old.sample_teacher(crop,origin,uv,np.ones((2,2),bool),operator)
        assert valid.all()
        close(value,103)
        bad,badvalid=old.sample_teacher(crop,origin,uv-1000,np.ones((2,2),bool),operator)
        assert not badvalid.any() and np.isnan(bad).all()
    # Non-four-aligned origins must still mean exact complete global4x4 blocks.
    yy,xx=np.mgrid[0:29,0:31]
    image=(xx+101)+10*(yy+203)
    pool,pool_origin=old.pooled_crop(image,(101,203))
    assert pool_origin==(26,51)
    close(pool[0,0],105.5+10*205.5)


def test_asymmetric_ramp_and_centroid():
    old=a.dependency()
    yy,xx=np.mgrid[0:80,0:80]
    image=3*(xx+100)+7*(yy+200)+11
    uv=np.array([[[28.3,53.1]],[[28.8,54.2]],[[30.1,55.3]]])
    mean=uv.mean(0)
    expected=3*(4*uv[...,0]+1.5)+7*(4*uv[...,1]+1.5)+11
    for operator in a.OPERATORS:
        values=[]
        for point in uv:
            v,m=old.sample_teacher(image,(100,200),point,np.ones((1,),bool),operator)
            assert m.all(); values.append(v)
        values=np.stack(values)
        close(values,expected)
        d,m=old.sample_teacher(image,(100,200),mean,np.ones((1,),bool),operator)
        close(values.mean(0),d)
        assert m.all()
    # Nonlinear teacher explicitly separates target averaging from centroid.
    quadratic=(xx+100)**2+2*(yy+200)**2
    vals=[old.sample_teacher(quadratic,(100,200),p,np.ones((1,),bool),'point_bilinear')[0] for p in uv]
    d,_=old.sample_teacher(quadratic,(100,200),mean,np.ones((1,),bool),'point_bilinear')
    assert np.max(np.abs(np.mean(vals,axis=0)-d))>1


def test_shift_sign():
    old=a.dependency()
    yy,xx=np.mgrid[:272,:272]
    q=np.zeros((20,272,272,3),np.float64)
    q[...,0]=xx+old.HALO_XY[0]
    q[...,1]=yy+old.HALO_XY[1]
    q[...,2]=np.arange(4,24)[:,None,None]
    valid=np.ones(q.shape[:-1],bool)
    bounds=(2416,1824,2448,1856)
    xy,m=old.shifted_queries(q,valid,bounds,14,(1,-2,3))
    close(xy[0,0],[2419,1822]);assert m.all()
    xyfit,_=old.shifted_queries(q,valid,bounds,14,a.FIT)
    close(xyfit[0,0],[2416,1823])


def test_equal_kernel_weights_and_moments():
    candidates=np.arange(5,dtype=np.float64)[:,None,None]*10
    indices=[[0,1],[2],[3],[4]]
    arms,means=a.operator_targets(candidates[0],candidates,candidates[0],indices[0],indices)
    close(arms['B'],5)
    close(arms['C'],(5+20+30+40)/4)
    assert not np.allclose(arms['C'],candidates.mean(0)) # detects equal-candidate bug
    close(means[:,0,0],[5,20,30,40])
    record=a.moments([(0,-1,0),(0,2,3)],[Fraction(1,3),Fraction(2,3)])
    assert record['mean_absolute_zyx_exact']==['0','1','2']
    assert record['mean_relative_to_fit_zyx_exact']==['0','2','2']
    assert record['covariance_zyx_exact']==[['0','0','0'],['0','2','2'],['0','2','2']]


def test_missing_support_and_no_zero_fill():
    shape=(2,4)
    raw=np.full(shape,60.0)
    c=np.stack([raw,raw])
    gc=np.ones(c.shape,bool); gc[1,0,0]=False
    sc=np.ones(c.shape,bool); sc[0,0,1]=False; c[0,0,1]=np.nan
    nd=np.ones(shape,bool);nd[0,2]=False
    d=raw.copy();d[0,3]=0
    one=np.ones(shape,bool)
    m=a.masks(raw,c,d,one,gc,one,one,sc,one,nd)
    assert int(m['geometry_only'].sum())==7
    assert int(m['source_only'].sum())==7
    assert int(m['complete'].sum())==5
    assert int(m['common'].sum())==4
    assert not m['common'][0].any() and m['common'][1].all()
    # Zero-filled invalid sampling is forbidden even when later masked.
    old=a.dependency()
    value,valid=old.bilinear(np.ones((4,4)),np.array([[[-1.0,2.0]]]),(0,0),np.ones((1,1),bool))
    assert not valid.any() and np.isnan(value).all()
    # Preserve a nonzero source value with geometry=false; mask excludes it.
    v,g=old.bilinear(np.ones((4,4))*70,np.array([[[1.0,2.0]]]),(0,0),np.zeros((1,1),bool))
    assert not g.any() and np.isnan(v).all()


def test_empty_positive_and_metrics():
    raw=np.array([[10.0,20],[30,40]])
    arms=dict(A=raw,B=raw+1,C=raw+2,D=raw+3)
    domains=dict(common=np.ones(raw.shape,bool),active_union=np.zeros(raw.shape,bool))
    s=a.summaries(arms,domains)
    assert s['differences']['A-B']['active_union']['MAE'] is None
    assert s['descriptive_quantization_comparisons']['B-D']['strictly_below_one_source_interval'] is None
    close(s['differences']['A-B']['common']['signed_mean'],-1/255)
    close(s['differences']['A-B']['common']['MAE'],1/255)
    close(a.entropy(np.array([0,.5,1])),[0,np.log(2),0])


def test_masked_bce_identity():
    logits=np.array([[-5.,-1.,.2],[1.1,3.2,8.]])
    targets=np.array([[[0.,.1,.3],[.5,.7,1.]],[[.1,.3,.8],[.9,.2,.6]],[[.2,.5,.7],[.1,.3,.8]]])
    mask=np.array([[True,False,True],[True,True,False]])
    bce=lambda q: np.logaddexp(0,logits)-logits*q
    lhs=np.mean([bce(q)[mask].mean() for q in targets])
    rhs=bce(targets.mean(axis=0))[mask].mean()
    close(lhs,rhs,tol=2e-15)
    weights=np.array([.25,.25,.5])
    weighted_lhs=sum(w*bce(q)[mask].mean() for w,q in zip(weights,targets))
    weighted_rhs=bce(np.einsum('i,iyx->yx',weights,targets))[mask].mean()
    close(weighted_lhs,weighted_rhs,tol=2e-15)


if __name__=='__main__':
    tests=[test_constant_and_origin,test_asymmetric_ramp_and_centroid,test_shift_sign,
           test_equal_kernel_weights_and_moments,test_missing_support_and_no_zero_fill,
           test_empty_positive_and_metrics,test_masked_bce_identity]
    results=[]
    for test in tests:
        test();results.append(dict(name=test.__name__,status='PASS'))
    a.save_json(a.HERE/'SYNTHETIC_TESTS.json',dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),
        test_groups=results,real_target_reads=0,analysis_sha256=a.sha(a.HERE/'analysis.py'),
        tests_sha256=a.sha(a.HERE/'synthetic_tests.py'),dependency_sha256=a.EXPECTED_DEPENDENCY))
    print(json.dumps(dict(status='PASS',groups=len(results))))
