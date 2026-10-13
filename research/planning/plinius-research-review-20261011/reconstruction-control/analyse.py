"""One-thread, fixed-plan local CT reconstruction comparison; no neural inference."""
import hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
VOLUMES=['20250728140407','20251107132835','20251107135911']

def pearson(a,b):
    if len(a)<2:return None
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    a=a-a.mean();b=b-b.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den else None

def spearman_uint8(a,b):
    ranks=[]
    for arr in [a,b]:
        counts=np.bincount(arr.ravel(),minlength=256)
        lut=np.cumsum(counts)-0.5*counts+0.5
        ranks.append(lut[arr.ravel()])
    return pearson(*ranks)

def phase_shift(a,b):
    if np.std(a)==0 or np.std(b)==0:return {'status':'flat-input','shift_to_apply_to_b_zyx':None}
    taper=np.ones(a.shape,dtype=np.float64)
    for axis,n in enumerate(a.shape):
        shp=[1]*a.ndim;shp[axis]=n;taper*=np.hanning(n).reshape(shp)
    fa=np.fft.rfftn((a.astype(float)-a.mean())*taper)
    fb=np.fft.rfftn((b.astype(float)-b.mean())*taper)
    cross=fa*np.conj(fb);mag=np.abs(cross)
    cross=np.divide(cross,mag,out=np.zeros_like(cross),where=mag>0)
    corr=np.fft.irfftn(cross,s=a.shape,axes=(0,1,2))
    idx=tuple(int(i) for i in np.unravel_index(np.argmax(corr),a.shape));peak=float(corr[idx])
    shift=[i if i<n//2 else i-n for i,n in zip(idx,a.shape)]
    # Exclude fixed radius2 neighbourhood with periodic indexing: diagnostic FFT grid only.
    neighbor=np.ix_(*[np.mod(np.arange(i-2,i+3),n) for i,n in zip(idx,a.shape)])
    far=corr.copy();far[neighbor]=-np.inf;second=float(np.max(far))
    tied=int(np.count_nonzero(np.isclose(corr,peak,rtol=1e-9,atol=1e-12)))
    return {'status':'unique-integer-peak' if tied==1 else 'tied-peak','shift_to_apply_to_b_zyx':shift,'peak':peak,'second_peak_outside_radius2':second,'peak_ratio':peak/second if second>0 else None,'equal_peak_count':tied,'scope':'Periodic integer diagnostic only; no subpixel or physical registration validation.'}

def compare(a,b):
    mask=(a>0)&(b>0);cnt=int(mask.sum());grad_a=[];grad_b=[]
    for axis in range(3):
        lo=[slice(None)]*3;hi=[slice(None)]*3;lo[axis]=slice(None,-1);hi[axis]=slice(1,None);lo=tuple(lo);hi=tuple(hi)
        good=mask[lo]&mask[hi]
        grad_a.append((a[hi].astype(np.int16)-a[lo].astype(np.int16))[good]);grad_b.append((b[hi].astype(np.int16)-b[lo].astype(np.int16))[good])
    result={'common_nonzero_count':cnt,'common_nonzero_fraction':cnt/a.size,'nonzero_mask_agreement':float(np.mean((a>0)==(b>0))),'gradient_common_edges':sum(len(x) for x in grad_a),'gradient_pearson_common_nonzero':pearson(np.concatenate(grad_a),np.concatenate(grad_b)),'phase_correlation':phase_shift(a,b)}
    for name,m in [('all',np.ones(a.shape,bool)),('common_nonzero',mask)]:
        aa=a[m];bb=b[m];result[name]={'count':len(aa),'pearson':pearson(aa,bb),'spearman':spearman_uint8(aa,bb) if len(aa)>1 else None,'mae_uint8':float(np.mean(np.abs(aa.astype(float)-bb))) if len(aa) else None}
    return result

def selfcheck():
    from scipy.stats import spearmanr
    rng=np.random.default_rng(71);a=rng.normal(size=(32,32,32));sh=(3,-2,1)
    got=phase_shift(a,np.roll(a,sh,axis=(0,1,2)))['shift_to_apply_to_b_zyx'];assert got==[-i for i in sh],got
    assert phase_shift(np.zeros((16,16,16)),np.ones((16,16,16)))['status']=='flat-input'
    x=np.array([0,0,1,3,3,255],np.uint8);y=np.array([7,4,4,0,255,255],np.uint8)
    assert abs(spearman_uint8(x,y)-spearmanr(x,y).statistic)<1e-12
    assert pearson(np.ones(4),np.ones(4)) is None
    assert abs(pearson(x,x)-1)<1e-12
    print(json.dumps({'checks':5,'passed':5,'known_roll_zyx':sh,'recovered_alignment_shift_zyx':got}))

def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plan=json.loads((BASE/'REQUEST_PLAN.json').read_text());ledger=json.loads((BASE/'REQUEST_LEDGER.json').read_text())
    assert len(ledger)==9 and all(x.get('status')==200 for x in ledger)
    cubes={};stats=[]
    for item in ledger:
        p=BASE/'chunks'/item['name'];body=p.read_bytes();assert len(body)==128**3 and hashlib.sha256(body).hexdigest()==item['sha256']
        a=np.frombuffer(body,np.uint8).reshape(128,128,128);cubes[item['site'],item['volume']]=a
        stats.append({'site':item['site'],'volume':item['volume'],'nonzero_fraction':float(np.mean(a>0)),'saturated255_fraction':float(np.mean(a==255)),'mean':float(a.mean()),'std':float(a.std()),'quantiles_0_1_10_50_90_99_100':np.percentile(a,[0,1,10,50,90,99,100]).tolist()})
    comparisons=[]
    for site in ['site1','site2','site3']:
        for v1,v2 in itertools.combinations(VOLUMES,2):
            result=compare(cubes[site,v1],cubes[site,v2]);comparisons.append({'site':site,'volume_a':v1,'volume_b':v2,**result})
            print(site,v1,v2,result['all']['pearson'],result['gradient_pearson_common_nonzero'],result['phase_correlation']['shift_to_apply_to_b_zyx'],flush=True)
        fig,axes=plt.subplots(3,3,figsize=(9,9),constrained_layout=True)
        for row,vid in enumerate(VOLUMES):
            a=cubes[site,vid]
            for col in range(3):
                ax=axes[row,col];ax.imshow(np.take(a,64,axis=col),cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_title(f'{vid} — {"ZYX"[col]}=64',fontsize=9);ax.set_axis_off()
        fig.suptitle(f'{site}: fixed orthogonal midplanes, uint8 0–255');fig.savefig(BASE/f'{site}-orthogonal.png',dpi=120);plt.close(fig)
    result={'protocol_sha256':hashlib.sha256((BASE/'PLAN.md').read_bytes()).hexdigest(),'request_plan_sha256':hashlib.sha256((BASE/'REQUEST_PLAN.json').read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Three fixed same-acquisition raw CT chunks, not surface/ink validation. No inference.','cubes':stats,'comparisons':comparisons}
    (BASE/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':selfcheck() if '--selfcheck' in sys.argv else main()
