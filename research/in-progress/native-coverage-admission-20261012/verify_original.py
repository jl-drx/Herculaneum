"""Independent dense-field/integral-mask replay of native support admission."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def integral(a):
    return np.pad(a.astype(np.uint32),((1,0),(1,0))).cumsum(0,dtype=np.uint32).cumsum(1,dtype=np.uint32)


def count(ii,x,y):
    return int(ii[y+128,x+128])-int(ii[y+128,x])-int(ii[y,x+128])+int(ii[y,x])


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=a.work/'original';result_path=a.work/'qualification/ORIGINAL_SUPPORT.csv'
    recorded=list(csv.DictReader(result_path.open()));rows_verified=complete=unknown=0;source_checks=0
    total_sup=total_pos=all21=0;candidate_keys=set()
    plan=json.loads((root/'REQUEST_PLAN.json').read_text())
    for winding in ['w035','w039','w040','w041','w044']:
        with np.load(root/'decoded'/f'{winding}.npz',allow_pickle=False) as f:
            mask=f['supervision'];target=f['target']
        assert mask.dtype==target.dtype==np.bool_ and not np.any(target&~mask)
        h,w=mask.shape;ii=integral(mask);expected=set()
        for y in range(0,h-127,96):
            for x in range(0,w-127,96):
                if count(ii,x,y)>=1024:expected.add((x,y))
        rr=[r for r in recorded if r['winding']==winding]
        assert expected=={(int(r['x']),int(r['y'])) for r in rr}
        del ii
        known=np.zeros((h,w),bool);valid=np.zeros((h,w),bool);depth_valid=np.zeros((h,w),bool)
        for r in plan['requests']:
            if r['source_id']!=winding:continue
            receipt=root/'receipts'/(r['id']+'.json')
            if not receipt.exists():continue
            rec=json.loads(receipt.read_text());assert rec['request']==r
            if rec['status']!='PRESENT':continue
            path=root/r['path'];assert path.stat().st_size==rec['bytes']==28*128*128 and sha(path)==rec['sha256']
            raw=np.fromfile(path,np.uint8).reshape(28,128,128);source_checks+=1
            y,x=r['cy']*128,r['cx']*128;hh,ww=min(128,h-y),min(128,w-x)
            known[y:y+hh,x:x+ww]=True
            valid[y:y+hh,x:x+ww]=raw[14,:hh,:ww]!=0
            depth_valid[y:y+hh,x:x+ww]=np.all(raw[4:25,:hh,:ww]!=0,axis=0)
        supported=mask&valid;positive=supported&target
        tables=[integral(v) for v in [known,supported,positive,supported&depth_valid]]
        for r in rr:
            x,y=int(r['x']),int(r['y']);values=[count(t,x,y) for t in tables]
            rows_verified+=1;candidate_keys.add(r['id'])
            if values[0]!=128**2:
                assert r['status']=='TRANSPORT_UNKNOWN'
                assert r['supervised_after_ct']==r['positive_after_ct']==r['negative_after_ct']==''
                assert r['technically_eligible']=='False';unknown+=1;continue
            assert r['status']=='CT_CHECKED'
            assert values[1:]==[int(r['supervised_after_ct']),int(r['positive_after_ct']),int(r['all21_nonzero_supervised'])]
            assert values[1]-values[2]==int(r['negative_after_ct'])
            assert (values[1]>=1024)==(r['technically_eligible']=='True')
            total_sup+=values[1];total_pos+=values[2];all21+=values[3];complete+=1
        del tables,known,valid,depth_valid,supported,positive,mask,target
    assert len(candidate_keys)==rows_verified==876
    result=dict(status='PASS',rows_verified=rows_verified,complete_ct_rows=complete,transport_unknown_rows=unknown,
        source_chunks_verified=source_checks,supervised_pixels_with_patch_overlap=total_sup,
        positive_pixels_with_patch_overlap=total_pos,supervised_pixels_with_all21_nonzero=all21,
        independent_method='Full-canvas known/centre/all21 CT fields and integral sums; no producer patch assembly or counting functions imported',
        result_csv_sha256=sha(result_path),implementation_sha256=sha(Path(__file__)),
        unknown_ct_not_scored=True,new_model_forwards=0,full_cohort_admitted=False,
        scope='Independent arithmetic in the same workspace; native annotation agreement/support only, not independent physical ink truth')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
