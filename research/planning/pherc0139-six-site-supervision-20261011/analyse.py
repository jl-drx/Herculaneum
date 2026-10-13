"""Offline target coverage, finite sensitivity and matched smoothing comparisons."""
from common import *
import argparse,csv

def metrics(values,fit,eligible,threshold):
    """Finite-set metrics; no assumption that teacher zero means physical absence."""
    assert np.isfinite(values).all() and np.isfinite(fit).all()
    nonzero=values>0;union=eligible & nonzero.any(0);common=eligible & nonzero.all(0)
    positive=(values>=threshold) & common;pu=positive.any(0);pc=positive.all(0);neg=common & ~pu
    variable=pu & ~pc;n=int(pu.sum());nc=int(common.sum());nu=int(union.sum())
    difference=np.abs(values-fit);active=values[:,pu];ranges=np.ptp(values,axis=0)
    worst=float(difference[:,pu].mean(1).max()/255) if n else None
    support=float((nu-nc)/nu) if nu else None;variation=float(variable.sum()/n) if n else None
    details=[]
    for value,diff in zip(values,difference):
        a=(fit>=threshold)&common;b=(value>=threshold)&common
        details.append(dict(gained_positive_pixels=int((~a & b).sum()),lost_positive_pixels=int((a & ~b).sum()),
            MAE_common_uint8=float(diff[common].mean()) if nc else None,
            MAE_positive_union_uint8=float(diff[pu].mean()) if n else None))
    result=dict(native_eligible_pixels=int(eligible.sum()),geometrically_complete_pixels=int(fit.size),
        teacher_nonzero_union_pixels=nu,teacher_nonzero_common_pixels=nc,teacher_support_variable_fraction=support,
        positive_union_pixels=n,positive_common_pixels=int(pc.sum()),negative_common_pixels=int(neg.sum()),
        variable_positive_pixels=int(variable.sum()),variable_positive_fraction=variation,
        worst_candidate_MAE_positive_union_normalized=worst,
        mean_score_range_common_uint8=float(ranges[common].mean()) if nc else None,
        p95_score_range_common_uint8=float(np.quantile(ranges[common],.95)) if nc else None,
        mean_score_range_positive_union_uint8=float(ranges[pu].mean()) if n else None,
        p95_score_range_positive_union_uint8=float(np.quantile(ranges[pu],.95)) if n else None,
        class_evidence_sufficient=bool(n>=100 and neg.sum()>=100),
        material_support=bool(support is not None and support>.01),
        material_hard_targets=bool(variation is not None and variation>.05),
        material_soft_targets=bool(worst is not None and worst>.05),per_candidate=details)
    result['material_any']=any(result[x] for x in ['material_support','material_hard_targets','material_soft_targets'])
    return result,common,pu,variable

def contrast(a,b,common,pu,threshold):
    d=np.abs(a-b)
    return dict(MAE_common_uint8=float(d[common].mean()) if common.any() else None,
        MAE_positive_union_uint8=float(d[pu].mean()) if pu.any() else None,
        max_absolute_uint8=float(d[common].max()) if common.any() else None,
        hard_disagreements_common=int(((a>=threshold)^(b>=threshold))[common].sum()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    verify(read(R/'INPUTS.json')['files']);verify(read(R/'REQUEST_PLAN.json')['code_and_inputs'])
    plan=read(R/'ALTERNATIVES.json');crops={r['core_id']:r for r in read(R/'teacher/CROPS.json')};rows=[]
    for site in plan['sites']:
        if not site['admitted_for_bounded_target_sensitivity']:continue
        sid=site['core_id'];d=R/'inputs/ct'/sid;meta=crops[sid];p=R/'teacher'/(sid+'.npy');assert sha(p)==meta['sha256']
        crop=np.load(p,allow_pickle=False);assert crop.dtype==np.uint8 and list(crop.shape)==meta['shape']
        q=np.load(d/'queries_uvk.npy',mmap_mode='r');support=np.load(d/'support.npy',mmap_mode='r');native=np.load(d/'native_full28.npy',mmap_mode='r');assert native.shape==(28,256,256)
        pool=T.pooled_crop(crop,meta['origin_xy']);folder=out/sid;folder.mkdir()
        for patch in site['patches']:
            x,y=patch['centre_core_xy'];eligible=native[:,y+64-12:y+64+12,x+64-12:x+64+12].max(0)>0
            for j in PLANES:
                qs=queries(q,support,patch,j,site['fit_translation_zyx'])
                for op in OPS:
                    vals={}
                    for name,uv in qs.items():
                        v,ok=T.sample_teacher(crop,meta['origin_xy'],uv,np.ones(uv.shape[:-1],bool),op,pool)
                        assert ok.all();vals[name]=v
                    values=vals['alternatives'];fit=vals['fit'];th=site['threshold'];m,common,pu,variable=metrics(values,fit,eligible,th)
                    avg=values.mean(0);stationary=vals['stationary'].mean(0)
                    record=dict(core_id=sid,winding=site['winding'],patch=patch['index'],native_plane=j,primary=j==14,operator=op,
                        threshold=th,alternatives=len(values),metrics=m,
                        controls=dict(mean_target_vs_mean_query=contrast(avg,vals['mean'],common,pu,th),
                            mean_target_vs_matched_stationary=contrast(avg,stationary,common,pu,th),
                            fit_vs_zero=contrast(fit,vals['zero'],common,pu,th)))
                    key=f'p{patch["index"]}-j{j:02d}-{op}'
                    np.savez_compressed(folder/(key+'.npz'),values=values,fit=fit,zero=vals['zero'],mean_query=vals['mean'],
                        mean_target=avg,stationary_target=stationary,eligible=eligible,common=common,positive_union=pu,variable_positive=variable)
                    rows.append(record)
        print(sid,'analysed',flush=True)
    summary=[]
    for op in OPS:
        for primary in [True,False]:
            r=[x for x in rows if x['operator']==op and (x['primary'] or not primary)];m=[x['metrics'] for x in r]
            summary.append(dict(operator=op,scope='plane14' if primary else 'all8planes',cases=len(r),
                sufficient=sum(x['class_evidence_sufficient'] for x in m),no_positive=sum(x['positive_union_pixels']==0 for x in m),
                hard_sensitive=sum(x['material_hard_targets'] for x in m),soft_sensitive=sum(x['material_soft_targets'] for x in m),
                support_sensitive=sum(x['material_support'] for x in m),sufficient_hard_sensitive=sum(x['class_evidence_sufficient'] and x['material_hard_targets'] for x in m),
                sufficient_soft_sensitive=sum(x['class_evidence_sufficient'] and x['material_soft_targets'] for x in m),
                pooled_positive_union=sum(x['positive_union_pixels'] for x in m),pooled_variable_positive=sum(x['variable_positive_pixels'] for x in m),
                max_stationary_control_MAE_common_uint8=max((x['controls']['mean_target_vs_matched_stationary']['MAE_common_uint8'] for x in r if x['metrics']['teacher_nonzero_common_pixels']),default=None),
                max_recenter_only_MAE_common_uint8=max((x['controls']['mean_target_vs_mean_query']['MAE_common_uint8'] for x in r if x['metrics']['teacher_nonzero_common_pixels']),default=None)))
    save(out/'RESULTS.json',dict(status='COMPLETE',scope='Conditional teacher-only sensitivity, not accuracy or paired-training admission',
        protocol_sha256=sha(R/'PROTOCOL.md'),request_plan_sha256=sha(R/'REQUEST_PLAN.json'),summary=summary,records=rows,
        retained_source_failures=plan['retained_source_failures']))
    with (out/'PRIMARY.csv').open('w',newline='') as f:
        fields=['core_id','patch','operator','alternatives','positive_union_pixels','negative_common_pixels','variable_positive_fraction','worst_candidate_MAE_positive_union_normalized','class_evidence_sufficient','material_hard_targets','material_soft_targets']
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:
            if r['primary']:w.writerow({k:(r[k] if k in r else r['metrics'][k]) for k in fields})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
