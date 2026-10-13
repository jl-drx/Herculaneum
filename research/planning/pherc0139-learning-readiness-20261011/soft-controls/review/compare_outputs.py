#!/usr/bin/env python3
"""Recompute saved-output scalar summaries independently and compare source replay."""
import argparse, json, hashlib
from fractions import Fraction
from pathlib import Path
import numpy as np
from independent_check import metrics, arm_stats

FIELD_MAP={'MAE':'mae','RMSE':'rmse','signed_mean':'signed_difference','p95_absolute':'p95_absolute_difference','max_absolute':'max_absolute_difference'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root, independent_dir=None, output_dir=None):
    independent_dir=independent_dir or root/'review';output_dir=output_dir or root/'review'
    results=root/'results';ind=json.loads((independent_dir/'INDEPENDENT_RESULTS.json').read_text());indrows={r['key']:r for r in ind['cases']};indarr=np.load(independent_dir/'independent_arrays.npz',allow_pickle=False)
    data=json.loads((results/'RESULTS.json').read_text());assert len(data['cases'])==64
    kernels=json.loads((results/'KERNELS.json').read_text());indices=[k['candidate_indices'] for k in kernels['local_kernels']]
    scalar_checks=0;max_scalar_error=0.;source_checks=[];threshold_differences=[];array_checks=0
    def equal(a,b,path):
        nonlocal scalar_checks,max_scalar_error
        scalar_checks+=1
        if a is None or b is None:assert a is b,(path,a,b);return
        if isinstance(a,(int,bool)) and isinstance(b,(int,bool)):assert a==b,(path,a,b);return
        d=abs(a-b);max_scalar_error=max(max_scalar_error,d);assert d<2e-14,(path,a,b,d)
    for r in data['cases']:
        p=results/r['arrays']['relative'];qp=results/r['queries']['relative'];assert sha(p)==r['arrays']['sha256'];assert sha(qp)==r['queries']['sha256']
        z=np.load(p,allow_pickle=False);q=np.load(qp,allow_pickle=False);key=f's{r["support"]}_j{r["native_plane"]}_{r["operator"]}';ir=indrows[key]
        raw=z['raw_candidates'];li=q['local_indices'];A=raw[list(map(tuple,q['shifts_zyx'])).index((0,-1,0))];B=raw[li].mean(0);km=np.stack([raw[k].mean(0) for k in indices]);C=km.mean(0);D=z['raw_D'];arms=dict(A=A,B=B,C=C,D=D)
        assert np.array_equal(km,z['all_kernel_means_raw'],equal_nan=True)
        for a,v in arms.items():assert np.array_equal(v,z['raw_'+a],equal_nan=True);assert np.array_equal(v/255,z['target_'+a],equal_nan=True);array_checks+=2
        geometry=q['fit_geometry_valid']&q['candidate_geometry_valid'].all(0)&q['centroid_geometry_valid'];source=z['source_fit_valid']&z['source_candidates_valid'].all(0)&z['source_centroid_valid'];finite=np.isfinite(A)&np.isfinite(raw).all(0)&np.isfinite(D);complete=geometry&source&finite&q['native_eligible'];nonzero=(A>0)&(raw>0).all(0)&(D>0);common=complete&nonzero;active=common&((A>=48)|(raw>=48).any(0)|(D>=48))
        masks=dict(geometry_only=geometry,source_only=source,finite=finite,native=q['native_eligible'],complete=complete,all_source_teacher_nonzero=nonzero,common=common,active_union=active)
        for k,v in masks.items():assert np.array_equal(v,z['mask_'+k]);equal(int(v.sum()),r['mask_counts'][k],key+'.mask.'+k);array_checks+=1
        equal(int((~source).sum()),r['outside_crop_pixels'],key+'.outside_crop_pixels');equal(int((~z['source_candidates_valid']).sum()+(~z['source_centroid_valid']).sum()),r['outside_crop_sample_occurrences'],key+'.outside_crop_sample_occurrences');equal(int((~geometry).sum()),r['geometry_missing_pixels'],key+'.geometry_missing_pixels')
        equal(int(common.sum()),ir['common_pixels'],key+'.independent_common');equal(int(active.sum()),ir['active_union_pixels'],key+'.independent_active')
        for a,v in arms.items():
            for domain,m in [('common',common),('active_union',active)]:
                s=arm_stats(v,m);actual=r['arms'][a][domain];equal(int(m.sum()),actual['pixels'],key+'.arm_pixels');equal(s['mean'],actual['target_mean'],key+'.mean');equal(s['entropy_nats'],actual['mean_target_entropy_nats'],key+'.entropy');equal(s['threshold48_positive_count'],actual['threshold48_positive_pixels'],key+'.positives')
        for a,b in [('A','B'),('A','C'),('B','C'),('B','D')]:
            for domain,m in [('common',common),('active_union',active)]:
                s=metrics(arms[a],arms[b],m);actual=r['differences'][a+'-'+b][domain]
                for f,ours in FIELD_MAP.items():equal(s[ours],actual[f],key+'.'+a+'-'+b+'.'+domain+'.'+f)
                equal(int(m.sum()),actual['pixels'],key+'.difference_pixels');la=arms[a][m]>=48;lb=arms[b][m]>=48;cnt=int((la^lb).sum());un=int((la|lb).sum());equal(cnt,actual['threshold48_disagreements'],key+'.disagreements');equal(un,actual['threshold48_positive_union_pixels'],key+'.union');equal(cnt/un if un else None,actual['threshold48_disagreement_fraction_of_pair_union'],key+'.disagreementfraction')
        for pair in ('B-C','B-D'):
            v=r['differences'][pair]['active_union']['MAE'];row=r['descriptive_quantization_comparisons'][pair];equal(v,row['active_union_MAE'],key+'.routingvalue');equal(1/255,row['source_quantization_interval'],key+'.quantinterval');equal(None if v is None else v<1/255,row['strictly_below_one_source_interval'],key+'.route');equal(None if v is None else v<1/255,ir['pairs'][pair]['active_union_mae_below_one_quantization_interval'],key+'.independentroute')
        source_max=0.;source_thresholds=0
        for a in ['A','B','C','D','candidates']:
            ours=indarr[key+('_transfers' if a=='candidates' else '_'+a)];their=z['raw_'+a]
            assert np.array_equal(np.isfinite(ours),np.isfinite(their)),(key,a,'finite mismatch')
            d=np.abs(ours-their);e=float(np.nanmax(d));source_max=max(source_max,e);assert e<1e-11,(key,a,e)
            changed=np.isfinite(ours)&((ours>=48)!=(their>=48));n=int(changed.sum());source_thresholds+=n
            if n:threshold_differences.append(dict(case=r['case'],array=a,pixels=n,independent_values=np.unique(ours[changed]).tolist(),production_values=np.unique(their[changed]).tolist(),primary=r['primary']))
        assert np.array_equal(common,indarr[key+'_common_mask']);assert np.array_equal(active,indarr[key+'_positive_union_mask']);assert np.array_equal(geometry,indarr[key+'_geometry_mask']);assert np.array_equal(complete,indarr[key+'_complete_mask'])
        assert np.array_equal(q['centroid_uv_group2'],indarr[key+'_centroid_uv'])
        source_checks.append(dict(case=r['case'],maximum_source_uint8_level_difference=source_max,raw_threshold_differences_counted_per_array=source_thresholds,mask_agreement=True,descriptive_decisions_agree=True))
    exact_checks=0
    shifts=kernels['unique_shifts_zyx']
    for rec in kernels['local_kernels']+[kernels['global_C']]:
        w=[Fraction(v) for v in rec['exact_weights_unique']];mu=[sum(p*s[d] for p,s in zip(w,shifts)) for d in range(3)]
        cov=[[sum(p*(s[d]-mu[d])*(s[e]-mu[e]) for p,s in zip(w,shifts)) for e in range(3)] for d in range(3)]
        assert [str(v) for v in mu]==rec['mean_absolute_zyx_exact'];assert [[str(v) for v in row] for row in cov]==rec['covariance_zyx_exact'];exact_checks+=12
    for r in json.loads((results/'CHECKSUMS.json').read_text())['files']:assert sha(results/r['relative'])==r['sha256']
    out=dict(status='PASS',cases=64,source_recomputed_cases=64,source_recomputed_candidate_arrays=64*24,scalar_checks=scalar_checks,array_checks=array_checks,exact_rational_moment_checks=exact_checks,maximum_scalar_error=max_scalar_error,maximum_source_transfer_uint8_level_difference=max(v['maximum_source_uint8_level_difference'] for v in source_checks),source_checks=source_checks,threshold_boundary_discrepancies=threshold_differences,decision_scope='Descriptive 1/255 comparisons only; no learning gate',production_result_sha256=sha(results/'RESULTS.json'),independent_result_sha256=sha(independent_dir/'INDEPENDENT_RESULTS.json'))
    (output_dir/'COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['source_checks','threshold_boundary_discrepancies']},indent=2));print('threshold discrepancy records',len(threshold_differences))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path('/workspace/herculaneum-lab/reports/pherc0139-soft-controls-20261011'));main(a.parse_args().root)
