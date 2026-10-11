"""Substantive numerical and fail-closed fixtures before target acquisition."""
from common import *
from analyse import metrics

def main():
    n=0
    # Query sign, depth origin, halo origin, X/Y order: non-symmetric synthetic field.
    z,y,x=np.mgrid[0:20,0:144,0:144];q=np.stack((10000*z+100*y+x,30000*z+500*y+7*x,np.zeros_like(x)),-1).astype(float)
    support=np.ones(q.shape[:-1],bool);center=(64,24);shift=[-2,3,-1]
    uv=query(q,support,center,14,shift);assert np.array_equal(uv,q[8,23:47,59:83,:2]);n+=1
    try:query(q,support,center,23,[6,0,0]);raise RuntimeError('failed to reject out-of-bounds')
    except AssertionError:n+=1
    support[8,23,59]=False
    try:query(q,support,center,14,shift);raise RuntimeError('failed to reject invalid geometry')
    except AssertionError:n+=1
    # Exact affine teacher across nonzero, nonmultiple-of-four crop origins.
    yy,xx=np.mgrid[11:70,7:81];crop=(xx+2*yy).astype(np.uint8);uv=np.array([[[5.17,6.29],[11.61,9.13]]]);valid=np.ones((1,2),bool)
    for op in OPS:
        v,ok=T.sample_teacher(crop,[7,11],uv,valid,op);expected=4*uv[...,0]+1.5+2*(4*uv[...,1]+1.5)
        assert ok.all() and np.max(np.abs(v-expected))<1e-12;n+=1
    # Nonlinear image: explicit scalar block means and interpolation for both operators.
    rng=np.random.default_rng(8841);crop=rng.integers(0,256,(59,74),dtype=np.uint8)
    for op in OPS:
        v,ok=T.sample_teacher(crop,[7,11],uv,valid,op)
        for idx in [(0,0),(0,1)]:
            u,w=uv[idx] if op==OPS[0] else 4*uv[idx]+1.5;a,b=int(np.floor(u)),int(np.floor(w));total=0.
            for dy in [0,1]:
                for dx in [0,1]:
                    value=float(crop[4*(b+dy)-11:4*(b+dy)+4-11,4*(a+dx)-7:4*(a+dx)+4-7].astype(np.int64).sum()/16) if op==OPS[0] else float(crop[b+dy-11,a+dx-7])
                    total+=value*(u-a if dx else 1-(u-a))*(w-b if dy else 1-(w-b))
            assert abs(v[idx]-total)<1e-12;n+=1
    # Threshold inclusivity, masked eligibility, class count and empty denominators.
    vals=np.array([[[48.,0.,47.]],[[47.,0.,48.]]]);m,*_=metrics(vals,vals[0],np.ones((1,3),bool),48)
    assert m['positive_union_pixels']==2 and m['variable_positive_fraction']==1 and not m['class_evidence_sufficient'];n+=1
    m,*_=metrics(np.zeros((2,2,2)),np.zeros((2,2)),np.ones((2,2),bool),48)
    assert m['variable_positive_fraction'] is None and m['worst_candidate_MAE_positive_union_normalized'] is None and m['teacher_support_variable_fraction'] is None;n+=1
    vals=np.full((2,24,24),60.);vals[:,:,12:]=30.;m,*_=metrics(vals,vals[0],np.ones((24,24),bool),48)
    assert m['class_evidence_sufficient'] and not m['material_any'];n+=1
    m,*_=metrics(vals,vals[0],np.zeros((24,24),bool),48);assert not m['class_evidence_sufficient'] and m['positive_union_pixels']==0;n+=1
    # A spatially constant displacement kernel must equal the original coordinates.
    base=rng.normal(size=(24,24,2));offset=rng.normal(size=(7,2));a=base[None]+offset[:,None,None,:];mean=a.mean(0);k=(a-mean).mean((1,2))
    assert np.max(np.abs(a-(mean[None]+k[:,None,None,:])))<1e-12;n+=1
    save(R/'SYNTHETIC_CHECKS.json',dict(status='PASS',checks=n,target_pixels_read=False))
    print('PASS',n,'synthetic checks')

if __name__=='__main__':main()
