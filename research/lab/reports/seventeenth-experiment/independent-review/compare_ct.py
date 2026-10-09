import hashlib
import json
import math
from pathlib import Path

P=Path('/workspace/herculaneum-lab/reports/seventeenth-experiment')
a=P/'implementation/ct-run/ct-diagnostics.json'
b=P/'independent-review/ct-reconstruction/independent-ct-diagnostics.json'
assert hashlib.sha256(a.read_bytes()).hexdigest()=='6e3930f01b2310f479c92d206cadd514e5b48fc7497ed7518b8634a670629fbf'
x=json.loads(a.read_text());y=json.loads(b.read_text());assert x['status']==y['status']=='PASS'
counts=dict(integer_fields=0,float_fields=0,null_fields=0,profiles=0,support_records=0)
max_abs={};max_rel={}
for ca,cb in zip(x['cases'],y['cases'],strict=True):
    assert ca['id']==cb['id']
    for key,n in cb['support_counts'].items():
        assert ca['support_counts'][key]==dict(coarse=n,expanded_native=9*n)
    for kind,total in [('native_profiles',51),('pooled_profiles',17)]:
        assert len(ca[kind])==len(cb[kind])==total
        for pa,pb in zip(ca[kind],cb[kind],strict=True):
            counts['profiles']+=1
            assert {k:v for k,v in pa.items()if k!='supports'}=={k:v for k,v in pb.items()if k!='supports'}
            assert set(pa['supports'])==set(pb['supports'])
            for name,sa in pa['supports'].items():
                counts['support_records']+=1;sb=pb['supports'][name];assert sa.keys()==sb.keys()
                for key,v in sa.items():
                    expected=sb[key]
                    if v is None or expected is None:assert v is expected;counts['null_fields']+=1
                    elif isinstance(v,int):assert type(expected)is int and v==expected;counts['integer_fields']+=1
                    else:
                        assert math.isfinite(v)and math.isfinite(expected)
                        difference=abs(v-expected);relative=difference/max(abs(expected),1e-300)
                        max_abs[key]=max(max_abs.get(key,0),difference);max_rel[key]=max(max_rel.get(key,0),relative)
                        assert math.isclose(v,expected,abs_tol=1e-9,rel_tol=1e-10),(ca['id'],kind,pa['plane_index'],name,key,v,expected)
                        counts['float_fields']+=1
    assert ca['exact_repooling']==dict(status='PASS',float32_voxels_checked=2506752,bitwise_equal=True)
    x0,y0,w,h=ca['bbox_source_xywh'];g=ca['coordinate_roundtrip']
    assert g['first_source_center_yx']==[y0+1,x0+1]and g['last_source_center_yx']==[y0+1150,x0+1150]
    assert g['source_plane_centers']==list(range(8,57,3)) and g['released_mapping_compared'] and not g['optical_registration_residual_measured']
    assert ca['cross_sections']==dict(coarse_center=192,native_local_column=577,native_local_row=577,nominal_spacing_um=3.24)
    pin=ca['saved_baseline_scores'];assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
assert counts['profiles']==408 and counts['support_records']==1632
r=dict(status='PASS',comparisons=counts,max_absolute_difference=max_abs,max_relative_difference=max_rel,
       tolerance=dict(atol=1e-9,rtol=1e-10),bitwise_pooled_voxels=15040512,
       input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [a,b]},
       implementation_review='Plot source uses frozen fixed planes, original coordinates, repeat3 masks, saved baseline score pins, row/column577 and source-table profiles. Root visual review remains separate.',
       models_executed=0,checkpoints_read=0)
with(P/'independent-review/ct-comparison-receipt.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r,indent=2))
