"""Independent numerical implementation; no import of study transfer/analysis."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,hashlib,json,zlib
import numpy as np
from scipy.ndimage import map_coordinates

def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sample(image,origin,uv,op):
    if op=='mean16_bilinear':
        ox,oy=origin;gx0=(ox+3)//4;gy0=(oy+3)//4;gx1=(ox+image.shape[1])//4;gy1=(oy+image.shape[0])//4
        # Explicit sixteen integer summands, independent of reshape/mean code.
        a=np.zeros((gy1-gy0,gx1-gx0),np.uint32)
        for dy in range(4):
            for dx in range(4):a+=image[4*gy0+dy-oy:4*gy1-oy:4,4*gx0+dx-ox:4*gx1-ox:4]
        a=a.astype(float)/16;coords=uv-np.array([gx0,gy0])
    else:a=image;coords=4*uv+1.5-np.array(origin)
    assert (coords[...,0]>=0).all() and (coords[...,0]<a.shape[1]-1).all() and (coords[...,1]>=0).all() and (coords[...,1]<a.shape[0]-1).all()
    return map_coordinates(a,[coords[...,1],coords[...,0]],order=1,prefilter=False,mode='constant',cval=np.nan,output=np.float64)

def reference_metrics(a,fit,e,t):
    full=e & (a>0).all(0);union=e & (a>0).any(0);p=(a>=t)&full
    pu=p.any(0);pc=p.all(0);nc=full & ~pu;changed=pu & ~pc
    n=int(pu.sum());nf=int(full.sum());nu=int(union.sum());d=np.abs(a-fit);ran=a.max(0)-a.min(0)
    support=(nu-nf)/nu if nu else None;hard=int(changed.sum())/n if n else None;soft=float(max(np.mean(b[pu]) for b in d)/255) if n else None
    result=dict(native_eligible_pixels=int(e.sum()),geometrically_complete_pixels=e.size,
        teacher_nonzero_union_pixels=nu,teacher_nonzero_common_pixels=nf,teacher_support_variable_fraction=support,
        positive_union_pixels=n,positive_common_pixels=int(pc.sum()),negative_common_pixels=int(nc.sum()),
        variable_positive_pixels=int(changed.sum()),variable_positive_fraction=hard,worst_candidate_MAE_positive_union_normalized=soft,
        mean_score_range_common_uint8=float(ran[full].mean()) if nf else None,p95_score_range_common_uint8=float(np.quantile(ran[full],.95)) if nf else None,
        mean_score_range_positive_union_uint8=float(ran[pu].mean()) if n else None,p95_score_range_positive_union_uint8=float(np.quantile(ran[pu],.95)) if n else None,
        class_evidence_sufficient=n>=100 and int(nc.sum())>=100,
        material_support=support is not None and support>.01,material_hard_targets=hard is not None and hard>.05,material_soft_targets=soft is not None and soft>.05)
    result['material_any']=any(result[k] for k in ['material_support','material_hard_targets','material_soft_targets'])
    result['per_candidate']=[]
    for b,delta in zip(a,d):
        bp=(b>=t)&full;fp=(fit>=t)&full
        result['per_candidate'].append(dict(gained_positive_pixels=int((bp & ~fp).sum()),lost_positive_pixels=int((fp & ~bp).sum()),
            MAE_common_uint8=float(delta[full].mean()) if nf else None,MAE_positive_union_uint8=float(delta[pu].mean()) if n else None))
    return result,full,pu,changed

def close(a,b):
    if isinstance(a,dict):assert set(a)==set(b);[close(a[k],b[k]) for k in a]
    elif isinstance(a,list):assert len(a)==len(b);[close(x,y) for x,y in zip(a,b)]
    elif isinstance(a,float):assert abs(a-b)<1e-9,(a,b)
    else:assert a==b,(a,b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--results',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args();r=args.study;out=args.results or r/'results'
    for e in read(r/'INPUTS.json')['files']+read(r/'REQUEST_PLAN.json')['code_and_inputs']:
        assert sha(r/e['path'])==e['sha256'] and (r/e['path']).stat().st_size==e['bytes'],e['path']
    ledger=read(r/'teacher/LEDGER.json');assert ledger['complete'] and ledger['request_plan_sha256']==sha(r/'REQUEST_PLAN.json')
    tiles={}
    for t in ledger['requests']:
        p=r/t['file'];assert sha(p)==t['compressed_sha256'];raw=zlib.decompress(p.read_bytes());assert len(raw)==1024**2 and hashlib.sha256(raw).hexdigest()==t['decoded_sha256']
        tiles[(t['winding'],*t['xy'])]=np.frombuffer(raw,np.uint8).reshape(1024,1024)
    ct=read(r/'inputs/ct/RESULTS.json');sites=read(r/'ALTERNATIVES.json')['sites'];records=read(out/'RESULTS.json')['records'];bykey={(x['core_id'],x['patch'],x['native_plane'],x['operator']):x for x in records}
    assert len(bykey)==len(records)
    for original in read(r/'inputs/geometry/NEXT_CT_PILOT_PLAN.json')['fixed_selected_sites']:
        core=original['scored_core_xywh'];context=original['native_context_xywh']
        assert core[:2]==[context[0]+64,context[1]+64] and core[2:]==[128,128] and context[2:]==[256,256]
    maxerror=0.;controls_error=0.;samples=0;cases=0;ties=[];source_perturbation=[]
    for s in sites:
        if not s['admitted_for_bounded_target_sensitivity']:continue
        sid=s['core_id'];w=s['winding'];x0,y0,x1,y1=s['source_bbox_fullfine_xyxy'];yy,xx=np.mgrid[y0:y1,x0:x1];crop=np.empty(yy.shape,np.uint8)
        for ty in range(y0//1024,(y1-1)//1024+1):
            for tx in range(x0//1024,(x1-1)//1024+1):
                mask=(yy//1024==ty)&(xx//1024==tx);crop[mask]=tiles[(w,tx,ty)][yy[mask]%1024,xx[mask]%1024]
        assert np.array_equal(crop,np.load(r/'teacher'/(sid+'.npy')))
        folder=r/'inputs/ct'/sid;q=np.load(folder/'queries_uvk.npy');support=np.load(folder/'support.npy');native=np.load(folder/'native_full28.npy');land=np.load(folder/'landscapes.npy')
        fit=s['fit_translation_zyx'];assert fit==next(z for z in ct['sites'] if z['core_id']==sid)['fit_translation_zyx']
        for patch in s['patches']:
            i=patch['index'];score=land[i];ix=tuple(z+6 for z in fit)
            chosen=[]
            for dz in range(-6,7):
                for dy in range(-6,7):
                    for dx in range(-6,7):
                        if score[dz+6,dy+6,dx+6]>=score[ix]-.05:chosen.append([dz,dy,dx])
            assert chosen==[c['shift_zyx'] for c in patch['alternatives']]
            cx,cy=patch['centre_core_xy'];ny,nx=np.mgrid[cy-12:cy+12,cx-12:cx+12]
            eligible=(native[:,ny+64,nx+64]!=0).any(0)
            for j in range(10,18):
                def direct(shift):
                    dz,dy,dx=shift;z=j+dz-4;y=ny+dy+8;x=nx+dx+8;assert support[z,y,x].all();return q[z,y,x,:2]
                uv=np.stack([direct(c) for c in chosen]);uf=direct(fit);uz=direct([0,0,0]);mean=np.sum(uv,axis=0)/len(uv)
                # Different reduction order: per-alternative spatial means minus their grand mean.
                means=uv.reshape(len(uv),-1,2).sum(axis=1)/576;k=means-means.sum(axis=0)/len(means);stationary=mean[None]+k[:,None,None,:]
                for op in ['mean16_bilinear','point_bilinear']:
                    item=bykey[(sid,i,j,op)];saved=np.load(out/sid/f'p{i}-j{j:02d}-{op}.npz');a=sample(crop,[x0,y0],uv,op);f=sample(crop,[x0,y0],uf,op);threshold=s['threshold']
                    err=float(np.max(np.abs(a-saved['values'])));assert err<1e-7;maxerror=max(maxerror,err);samples+=a.size
                    assert np.array_equal(eligible,saved['eligible'])
                    # All reported metrics are independently calculated from exactly saved values.
                    m,common,pu,var=reference_metrics(saved['values'],saved['fit'],eligible,threshold);close(m,item['metrics'])
                    assert np.array_equal(common,saved['common']) and np.array_equal(pu,saved['positive_union']) and np.array_equal(var,saved['variable_positive'])
                    original=saved['values'];mismatch=(a>=threshold)!=(original>=threshold)
                    if mismatch.any():
                        assert np.max(np.abs(a[mismatch]-threshold))<1e-7 and np.max(np.abs(original[mismatch]-threshold))<1e-7
                        ties.append(dict(core_id=sid,patch=i,plane=j,operator=op,count=int(mismatch.sum()),indices=np.argwhere(mismatch).tolist()))
                    sm,_,_,_=reference_metrics(a,f,eligible,threshold)
                    changes={k:[item['metrics'][k],sm[k]] for k in ['positive_union_pixels','variable_positive_pixels','class_evidence_sufficient','material_hard_targets','material_soft_targets','material_support'] if item['metrics'][k]!=sm[k]}
                    if changes:source_perturbation.append(dict(core_id=sid,patch=i,plane=j,operator=op,changes=changes))
                    recompute={'fit':f,'zero':sample(crop,[x0,y0],uz,op),'mean_query':sample(crop,[x0,y0],mean,op),'mean_target':a.mean(0),'stationary_target':sample(crop,[x0,y0],stationary,op).mean(0)}
                    for name,value in recompute.items():
                        error=float(np.max(np.abs(value-saved[name])));assert error<1e-7;controls_error=max(controls_error,error)
                    for label,an,bn in [('mean_target_vs_mean_query','mean_target','mean_query'),('mean_target_vs_matched_stationary','mean_target','stationary_target'),('fit_vs_zero','fit','zero')]:
                        av,bv=saved[an],saved[bn];d=np.abs(av-bv);m=dict(MAE_common_uint8=float(d[common].mean()) if common.any() else None,
                            MAE_positive_union_uint8=float(d[pu].mean()) if pu.any() else None,max_absolute_uint8=float(d[common].max()) if common.any() else None,
                            hard_disagreements_common=int(((av>=threshold)^(bv>=threshold))[common].sum()))
                        close(m,item['controls'][label])
                    cases+=1
        print(sid,'independent source transfer/metrics PASS',flush=True)
    assert cases==len(records)==384
    # Recompute aggregate rows rather than merely trusting per-case calculations.
    for summary in read(out/'RESULTS.json')['summary']:
        rows=[x for x in records if x['operator']==summary['operator'] and (summary['scope']=='all8planes' or x['primary'])];m=[x['metrics'] for x in rows]
        expected=dict(operator=summary['operator'],scope=summary['scope'],cases=len(rows),
            sufficient=sum(x['class_evidence_sufficient'] for x in m),no_positive=sum(x['positive_union_pixels']==0 for x in m),
            hard_sensitive=sum(x['material_hard_targets'] for x in m),soft_sensitive=sum(x['material_soft_targets'] for x in m),support_sensitive=sum(x['material_support'] for x in m),
            sufficient_hard_sensitive=sum(x['class_evidence_sufficient'] and x['material_hard_targets'] for x in m),sufficient_soft_sensitive=sum(x['class_evidence_sufficient'] and x['material_soft_targets'] for x in m),
            pooled_positive_union=sum(x['positive_union_pixels'] for x in m),pooled_variable_positive=sum(x['variable_positive_pixels'] for x in m),
            max_stationary_control_MAE_common_uint8=max(x['controls']['mean_target_vs_matched_stationary']['MAE_common_uint8'] for x in rows if x['metrics']['teacher_nonzero_common_pixels']),
            max_recenter_only_MAE_common_uint8=max(x['controls']['mean_target_vs_mean_query']['MAE_common_uint8'] for x in rows if x['metrics']['teacher_nonzero_common_pixels']))
        close(expected,summary)
    assert not any(any(k.startswith('material_') or k=='class_evidence_sufficient' for k in c['changes']) for c in source_perturbation),'Routing changed under alternate arithmetic'
    report=dict(status='PASS',scope='Different numerical implementations in the same runtime, not an independent team or physical validation',
        cases=cases,source_alternative_samples=samples,max_source_transfer_error_uint8=maxerror,max_control_error_uint8=controls_error,
        all_saved_metrics_recomputed=True,all_routing_decisions_preserved=True,threshold_ties=ties,alternate_arithmetic_count_changes=source_perturbation)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['threshold_ties','alternate_arithmetic_count_changes']}))

if __name__=='__main__':main()
