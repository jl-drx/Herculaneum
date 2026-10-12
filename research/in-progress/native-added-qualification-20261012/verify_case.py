"""Independent full-raster CT assembly and saved teacher/support accounting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    q=a.qualification
    result=json.loads((q/'RESULTS.json').read_text())
    freeze=json.loads((q/'FREEZE.json').read_text())
    plan=json.loads((a.work/'REQUEST_PLAN.json').read_text())
    assert sha(q/'FREEZE.json')==result['freeze_sha256']
    assert sha(a.work/'REQUEST_PLAN.json')==freeze['request_plan_sha256']
    assert sha(q/'qualify_added.py')==freeze['implementation_sha256']
    for name,digest in freeze['reference_sha256'].items():assert sha(q/'reference'/name)==digest
    verified=[]
    for outcome in result['cases']:
        if not outcome['supervision_qualified']:continue
        ident=outcome['id'];case=next(r for r in plan['cases'] if r['id']==ident)
        d,h,w=case['native_shape'];assert d==28
        volume=np.zeros((d,h,w),np.uint8);covered=np.zeros((h,w),bool);chunks=0
        for r in plan['requests']:
            if r['source_id']!=ident:continue
            rp=a.work/'receipts'/(r['id']+'.json');rec=json.loads(rp.read_text())
            assert sha(rp)==freeze['source_receipt_sha256'][r['id']]
            assert rec['request']==r and rec['status']=='PRESENT'
            assert sha(a.work/r['path'])==rec['sha256']
            if r['kind']!='ct':continue
            raw=np.fromfile(a.work/r['path'],np.uint8).reshape(28,128,128)
            y=r['cy']*128;x=r['cx']*128;hh=min(128,h-y);ww=min(128,w-x)
            assert hh>0 and ww>0
            volume[:,y:y+hh,x:x+ww]=raw[:,:hh,:ww];covered[y:y+hh,x:x+ww]=True;chunks+=1
        for name,digest in outcome['outputs'].items():assert sha(q/name)==digest
        rows=list(csv.DictReader((q/(ident+'-support.csv')).open()))
        originals=[r for r in csv.DictReader((q/'reference/CANDIDATES.csv').open()) if r['segment']==case['segment']]
        assert len(rows)==len(originals)==case['candidates']
        with np.load(q/(ident+'-patches.npz'),allow_pickle=False) as f:
            teachers=f['teacher'];valids=f['valid']
        assert teachers.shape==valids.shape==(len(rows),128,128)
        # -1 means not observed. Never score outside retained candidate support.
        label_map=np.full((h,w),-1,np.int16)
        unique={name:np.zeros((h,w),bool) for name in ['restricted_A','broad_B']}
        totals={name:dict(candidates=0,technically_eligible=0,zero_positive_patches=0,zero_negative_patches=0,supervised_occurrences=0,positive_occurrences=0) for name in unique}
        depth_checks=0
        for i,(row,original) in enumerate(zip(rows,originals,strict=True)):
            for key,value in original.items():assert row[key]==value
            y=int(row['y']);x=int(row['x']);region=np.s_[y:y+128,x:x+128]
            assert covered[region].all()
            native=volume[:,y:y+128,x:x+128]
            valid=native[14]!=0
            assert np.array_equal(valid,valids[i])
            teacher=teachers[i]
            old=label_map[region]
            assert np.all((old==-1)|(old==teacher))
            label_map[region]=teacher
            mask=valid & ((teacher<=64)|(teacher>=128))
            positive=int(np.count_nonzero(mask & (teacher>=128)))
            supervised=int(np.count_nonzero(mask));negative=supervised-positive
            assert int(row['supervised'])==supervised and int(row['positive'])==positive and int(row['negative'])==negative
            assert int(row['ignored'])==16384-supervised
            assert int(row['valid_ct_pixels'])==int(valid.sum())
            assert int(row['intermediate_teacher_pixels'])==int(((teacher>64)&(teacher<128)).sum())
            assert int(row['supervision_removed_by_ct'])==int((~valid & ((teacher<=64)|(teacher>=128))).sum())
            assert (row['technically_eligible']=='True')==(supervised>=1024)
            for layer in range(28):
                assert int(row[f'ct_nonzero_depth_{layer:02}'])==int(np.count_nonzero(native[layer]));depth_checks+=1
            for name in ['broad_B']+(['restricted_A'] if int(row['in_restricted_A']) else []):
                t=totals[name];t['candidates']+=1;t['technically_eligible']+=supervised>=1024
                t['zero_positive_patches']+=positive==0;t['zero_negative_patches']+=negative==0
                t['supervised_occurrences']+=supervised;t['positive_occurrences']+=positive
                unique[name][region] |= mask
        for expected in outcome['counts']:
            name=expected['arm'];actual=dict(arm=name,**totals[name],unique_supervised_uv=int(unique[name].sum()),unique_positive_uv=int((unique[name] & (label_map>=128)).sum()))
            assert actual==expected,(actual,expected)
        verified.append(dict(id=ident,candidates=len(rows),ct_chunks=chunks,depth_counts_verified=depth_checks,
            candidate_ancestry_equal=True,saved_ct_support_equal=True,overlapping_teacher_pixels_equal=True,
            class_and_unique_uv_counts_equal=True))
        del volume,covered,label_map,unique,teachers,valids
    receipt=dict(status='PASS',cases=verified,implementation_sha256=sha(Path(__file__)),
        qualification_results_sha256=sha(q/'RESULTS.json'),
        scope='Independent full-raster CT assembly and accounting; TIFF/area agreement belongs to the separate production qualification. Not physical registration or accuracy.',
        full_cohort_admitted=False,new_model_forwards=0,paid_compute=False)
    a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
