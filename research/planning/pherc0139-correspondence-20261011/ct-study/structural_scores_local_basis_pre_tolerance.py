"""Finite translation NCC. No acquisition, model, optimised warp or display tuning."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter

FIT_XY=[(48,48),(208,48),(48,208),(208,208)]
CHECK_XY=[(128,48),(48,128),(208,128),(128,208)]
SHIFTS=np.moveaxis(np.indices((13,13,13)),0,-1).astype(np.int16)-6

def window_sum(x, shape):
    s=np.pad(np.asarray(x,dtype=np.float64),((1,0),(1,0),(1,0)))
    s=s.cumsum(0).cumsum(1).cumsum(2)
    a,b,c=shape
    return (s[a:,b:,c:]-s[:-a,b:,c:]-s[a:,:-b,c:]-s[a:,b:,:-c]
            +s[:-a,:-b,c:]+s[:-a,b:,:-c]+s[a:,:-b,:-c]-s[:-a,:-b,:-c])

def ncc_landscape(search,template):
    search=np.asarray(search,dtype=np.float64); template=np.asarray(template,dtype=np.float64)
    if not (np.isfinite(search).all() and np.isfinite(template).all()):
        raise ValueError('Incomplete/nonfinite support')
    tc=template-template.mean(); var_t=np.sum(tc*tc)
    if var_t<=1e-12: raise ValueError('Constant template')
    sums=window_sum(search,template.shape)
    sumsq=window_sum(search*search,template.shape)
    var_s=np.maximum(sumsq-sums*sums/template.size,0)
    numer=correlate(search,tc,mode='valid',method='fft')
    result=np.full(var_s.shape,np.nan)
    good=var_s>1e-12
    result[good]=numer[good]/np.sqrt(var_t*var_s[good])
    if np.isfinite(result).any() and np.nanmax(np.abs(result))>1+1e-7: raise ValueError('NCC arithmetic out of range')
    return np.clip(result,-1,1)

def direct_ncc(a,b):
    a=a.ravel().astype(float);b=b.ravel().astype(float)
    a-=a.mean();b-=b.mean()
    return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))

def peak(scores):
    if not np.isfinite(scores).all(): raise ValueError('Unscorable translation')
    idx=np.unravel_index(np.argmax(scores),scores.shape)
    return tuple(int(x-6) for x in idx),float(scores[idx])

def landscapes(native,fine):
    """native/fine shape20,272,272; first layer corresponds to global j4."""
    assert native.shape==fine.shape==(20,272,272)
    scores=[]
    for x,y in FIT_XY+CHECK_XY:
        # Core zero is halo8; native j10:18 has array indices6:14.
        y0=y+8-16; x0=x+8-16
        template=native[6:14,y0:y0+32,x0:x0+32]
        search=fine[:,y0-6:y0+38,x0-6:x0+38]
        score=ncc_landscape(search,template)
        assert score.shape==(13,13,13)
        scores.append(score)
    return np.stack(scores)

def summarise(scores,basis,local_bases=None):
    scores=np.asarray(scores);basis=np.asarray(basis) # columns native XYZ from delta(Z,Y,X)
    meanfit=scores[:4].mean(0); fitshift,fitvalue=peak(meanfit)
    fitidx=tuple(x+6 for x in fitshift); fs=np.array(fitshift)
    if local_bases is None: local_bases=[basis]*len(scores)
    assert len(local_bases)==len(scores)
    entries=[]
    for i,score in enumerate(scores):
        local=np.asarray(local_bases[i])
        pdelta=(SHIFTS-fs)@local.T
        dist=np.linalg.norm(pdelta,axis=-1)
        n=local[:,0]/np.linalg.norm(local[:,0]); far=dist>2.0
        normalfar=np.abs(pdelta@n)>2.0
        best,val=peak(score); delta=local@(np.array(best)-fs)
        dn=float(delta@n);dt=float(np.linalg.norm(delta-dn*n))
        comp=np.where(far,score,-np.inf); ci=np.unravel_index(np.argmax(comp),comp.shape)
        nc=np.where(normalfar,score,-np.inf); ni=np.unravel_index(np.argmax(nc),nc.shape)
        entries.append(dict(index=i,role='fit' if i<4 else 'check',centre_core_xy=list((FIT_XY+CHECK_XY)[i]),
            zero_score=float(score[6,6,6]),at_fit_score=float(score[fitidx]),independent_best_zyx=list(best),independent_best_score=val,
            independent_peak_on_boundary=any(abs(s)==6 for s in best),check_fit_normal_delta_native_vox=dn,
            check_fit_tangential_delta_native_vox=dt,check_fit_distance_native_vox=float(np.linalg.norm(delta)),
            distinct_competitor_zyx=[int(v-6) for v in ci],distinct_competitor_score=float(score[ci]),
            margin=float(score[fitidx]-score[ci]),normal_competitor_zyx=[int(v-6) for v in ni],
            normal_competitor_score=float(score[ni])))
    checks=entries[4:]
    criteria=dict(fit_not_boundary=not any(abs(s)==6 for s in fitshift),checks_not_boundary=all(not e['independent_peak_on_boundary'] for e in checks),
        all_check_ncc_at_least_05=all(e['at_fit_score']>=.5 for e in checks),all_check_margin_at_least_005=all(e['margin']>=.05 for e in checks),
        median_tangent_delta_at_most_05=float(np.median([e['check_fit_tangential_delta_native_vox'] for e in checks]))<=.5,
        max_tangent_delta_at_most_1=max(e['check_fit_tangential_delta_native_vox'] for e in checks)<=1,
        max_abs_normal_delta_at_most_1=max(abs(e['check_fit_normal_delta_native_vox']) for e in checks)<=1)
    fitxyz=basis@fs
    return dict(fit_translation_zyx=list(fitshift),fit_mean_ncc=fitvalue,baseline_mean_fit_ncc=float(meanfit[6,6,6]),
        fit_translation_native_xyz=fitxyz.tolist(),fit_translation_micrometres=(fitxyz*9.362).tolist(),
        criteria=criteria,structural_screen_pass=all(criteria.values()),supports=entries)

def fixtures():
    rng=np.random.default_rng(139040)
    f=gaussian_filter(rng.normal(size=(32,96,96)),sigma=(.7,1.,1.))
    out=[]
    for shift in [(0,0,0),(2,-3,4),(-4,3,-2),(0,0,6)]:
        z,y,x=shift
        t=2.3*f[12+z:20+z,32+y:64+y,32+x:64+x]+17
        search=f[6:26,26:70,26:70]
        scores=ncc_landscape(search,t); estimated,value=peak(scores)
        assert estimated==shift and abs(value-1)<1e-10
        for idx in [(0,0,0),(6,6,6),(12,12,12),(4,7,9)]:
            zz,yy,xx=idx; brute=direct_ncc(search[zz:zz+8,yy:yy+32,xx:xx+32],t)
            assert abs(brute-scores[idx])<1e-10
        out.append(dict(true_zyx=shift,recovered_zyx=estimated,peak=value))
    repeated=np.tile(rng.normal(size=(4,44,44)),(5,1,1))
    repeated=gaussian_filter(repeated,sigma=(0,1,1))
    t=repeated[6:14,6:38,6:38]
    periodic=ncc_landscape(repeated,t)
    peaks=[int(z-6) for z in range(13) if abs(periodic[z,6,6]-1)<1e-10]
    assert len(peaks)>=3
    constant_rejected=False
    try: ncc_landscape(np.ones((20,44,44)),np.ones((8,32,32)))
    except ValueError: constant_rejected=True
    assert constant_rejected
    return dict(status='PASS',known_shift_cases=out,periodic_equal_depth_peaks=peaks,constant_template_rejected=constant_rejected,
                limitation='Synthetic finite-translation arithmetic check, not physical registration sensitivity or ink truth.')

if __name__=='__main__':
    result=fixtures();p=Path(__file__).with_name('FIXTURES.json')
    with p.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
