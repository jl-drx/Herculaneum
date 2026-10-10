"""Reconstruct saved tile sums independently and describe all fixed offsets."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
W=P/'corrected-run'
run=json.loads((W/'RUN_RESULT.json').read_text());assert run['status']=='PASS'
plans=json.loads((P/'SCHEDULE.json').read_text())
go=json.loads((P/'GO_V3.json').read_text())
for r in go['pins']:
    with Path(r['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
y,x,Y,X=go['roi_yxYX'];shape=(Y-y,X-x)
valid=np.load(P.parent/'fortysixth-experiment/work/valid.npy')[y:Y,x:X]
weights=np.load(W/'blend_weights.npy')
maps=[];checks=[]
for rec,plan in zip(run['offsets'],plans):
    out=Path(rec['response']['path']).parent
    total=np.zeros(shape,np.float32);mass=np.zeros(shape,np.float32)
    assert len(rec['batches'])==len(plan['retained_batches'])
    for batch,pin in zip(plan['retained_batches'],rec['batches']):
        p=Path(pin['probabilities']['path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['probabilities']['sha256']
        tiles=np.load(p)
        # In this admitted fixed ROI all retained cohorts were raw-nonempty.
        # A differing count is not silently aligned to the first N block IDs.
        assert len(tiles)==len(batch)==pin['batch']
        for blockid,tile in zip(batch,tiles[:,0]):
            yy,xx,h,w=plan['blocks'][blockid]
            ya,xa,yb,xb=max(y,yy),max(x,xx),min(Y,yy+h),min(X,xx+w)
            if ya>=yb or xa>=xb:continue
            src=np.s_[ya-yy:yb-yy,xa-xx:xb-xx];dst=np.s_[ya-y:yb-y,xa-x:xb-x]
            total[dst]+=tile[src]*weights[src];mass[dst]+=weights[src]
    assert np.array_equal(total,np.load(out/'probability_sum.npy'))
    assert np.array_equal(mass,np.load(out/'weight_sum.npy'))
    assert (mass[valid]>1e-6).all()
    np.divide(total,mass,out=total,where=mass>1e-6);total=np.clip(total,0,1);total*=255
    actual=np.load(out/'response.npy')
    assert np.array_equal(total.astype(np.uint8),actual)
    maps.append(actual);checks.append({'offset':rec['offset_yx'],'independent_sum_weight_and_uint8_exact':True})

Yg,Xg=np.indices(shape);Yg+=y;Xg+=x

def rank(im,q):
    vals=im[valid];k=int(np.ceil(q*len(vals)));cut=np.partition(vals,len(vals)-k)[len(vals)-k]
    b=valid&(im>=cut)
    return b,{'threshold':int(cut),'selected':int(b.sum()),'actual_fraction':float(b.sum()/valid.sum())}

results=[]
for i,j in itertools.combinations(range(4),2):
    aa,bb=maps[i][valid].astype(float),maps[j][valid].astype(float)
    d=np.abs(maps[i].astype(np.int16)-maps[j].astype(np.int16))
    rec={'offsets':[plans[i]['offset_yx'],plans[j]['offset_yx']],
         'mae':float(d[valid].mean()),'max':int(d[valid].max()),'pearson':float(np.corrcoef(aa,bb)[0,1]),'ranks':[]}
    for q in [.05,.1,.2]:
        a,ma=rank(maps[i],q);b,mb=rank(maps[j],q)
        da=distance_transform_edt(~b)[a];db=distance_transform_edt(~a)[b];dd=np.r_[da,db]
        # Independent KD-tree verifies every distance for these small supports.
        pa,pb=np.argwhere(a),np.argwhere(b)
        assert np.array_equal(da,cKDTree(pb).query(pa,workers=1)[0])
        assert np.array_equal(db,cKDTree(pa).query(pb,workers=1)[0])
        rec['ranks'].append({'q':q,'selections':[ma,mb],'dice':float(2*(a&b).sum()/(a.sum()+b.sum())),
            'within_radius':{str(r):{'A_to_B':float((da<=r).mean()),'B_to_A':float((db<=r).mean()),'pooled':float((dd<=r).mean())} for r in [0,2,4,8]},
            'nearest_mean':float(dd.mean()),'nearest_p95':float(np.percentile(dd,95))})
    if i==0:
        rec['phase8x8']={}
        for label,(dy,dx) in [('source',(0,0)),('baseline',plans[i]['offset_yx']),('shifted',plans[j]['offset_yx'])]:
            phase=(((Yg-dy)%64)//8)*8+((Xg-dx)%64)//8
            count=np.bincount(phase[valid],minlength=64)
            sums=np.bincount(phase[valid],weights=d[valid],minlength=64)
            rec['phase8x8'][label]={'counts':count.tolist(),'mae':np.divide(sums,count,out=np.full(64,np.nan),where=count>0).tolist()}
    results.append(rec)

fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
for k,(im,plan) in enumerate(zip(maps,plans)):
    display=np.clip((im.astype(float)-64)/128,0,1)
    ax=axes[0,k];ax.imshow(np.ma.array(display,mask=~valid),cmap='gray',vmin=0,vmax=1);ax.set_title(str(tuple(plan['offset_yx']))+' source aligned')
    if k==0:
        selected=np.stack([rank(m,.1)[0] for m in maps]).sum(0)
        axes[1,k].imshow(np.ma.array(selected,mask=~valid),vmin=0,vmax=4,cmap='viridis');axes[1,k].set_title('Top10% selection count,0–4')
    else:
        diff=np.abs(im.astype(np.int16)-maps[0].astype(np.int16))
        axes[1,k].imshow(np.ma.array(diff,mask=~valid),vmin=0,vmax=64,cmap='magma');axes[1,k].set_title('Absolute difference from(0,0),0–64')
for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('One fixed exposed PHerc343 ROI; native source orientation; fixed display scales\nGrid sensitivity is not ink accuracy; all four predetermined offsets retained')
fig.savefig(P/'comparison.png',dpi=130);plt.close(fig)
report={'status':'PASS','valid_pixels':int(valid.sum()),'total_roi_pixels':valid.size,'baseline_gate':run['offsets'][0]['baseline_gate'],
        'independent_reconstruction':checks,'distance_verification':'All18 pair/rank bidirectional distance arrays agree exactly between EDT and KD-tree.',
        'comparisons':results,'new_corrected_tile_forwards':sum(r['tile_forwards'] for r in run['offsets']),
        'failed_attempt_tile_forwards':108,'scope':'One exposed ROI, source-aligned operational grid sensitivity; no ink accuracy, causal mechanism identification or phase selection.'}
with (P/'ANALYSIS.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':'PASS','valid_pixels':report['valid_pixels'],'tile_forwards':report['new_corrected_tile_forwards'],
                 'pairs':[{k:v for k,v in r.items() if k not in ['ranks','phase8x8']} for r in results]},indent=2))
