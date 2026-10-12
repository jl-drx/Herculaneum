"""Offline dependency accounting and native annotation/CT support qualification.

An unresolved transport dependency prevents admission. It never becomes a zero
CT image, a blank target, or a reason to replace the candidate.
"""
import argparse
import collections
import csv
import functools
import hashlib
import json
from pathlib import Path
import numpy as np
from intake import sha, stamp, write
from original_stream import csv_write


def receipts(root):
    plan=json.loads((root/'REQUEST_PLAN.json').read_text())
    found={};summary=collections.Counter()
    for r in plan['requests']:
        path=root/'receipts'/(r['id']+'.json')
        if not path.exists():summary['unresolved_or_unattempted']+=1;continue
        rec=json.loads(path.read_text());assert rec['request']==r
        if rec['status']=='PRESENT':
            source=root/r['path'];assert source.stat().st_size==r['bytes']==rec['bytes'] and sha(source)==rec['sha256']
            summary['verified_present']+=1
        else:assert rec['status']=='MISSING';summary['confirmed_missing']+=1
        found[r['id']]=rec
    return plan,found,dict(summary)


def source_status(required, found):
    present=sum(key in found and found[key]['status']=='PRESENT' for key in required)
    missing=sum(key in found and found[key]['status']=='MISSING' for key in required)
    unknown=len(required)-present-missing
    return dict(required=len(required),present=present,confirmed_missing=missing,unresolved=unknown)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    out=a.work/'qualification';out.mkdir(exist_ok=False)
    plan,found,source_counts=receipts(a.work)
    rows=list(csv.DictReader((a.base/'support-reservation-v2/CANDIDATES.csv').open()))
    cases={s['segment']:s for s in plan['cases']};results=[]
    for row in rows:
        case=cases[row['segment']];ident=case['id'];x,y=int(row['x']),int(row['y'])
        ct=[f'{ident}-ct-y{cy:04}-x{cx:04}' for cy in range(y//128,(y+127)//128+1) for cx in range(x//128,(x+127)//128+1)]
        _,h,w=case['native_shape'];fh,fw=case['teacher_shape'];columns=(fw+1023)//1024
        x0=max(0,x*fw//w-2);x1=min(fw,((x+128)*fw+w-1)//w+2)
        y0=max(0,y*fh//h-2);y1=min(fh,((y+128)*fh+h-1)//h+2)
        teacher=[f'{ident}-teacher-{ty*columns+tx:05}' for ty in range(y0//1024,(y1-1)//1024+1) for tx in range(x0//1024,(x1-1)//1024+1)]
        c=source_status(ct,found);t=source_status(teacher,found)
        status='SOURCE_MISSING' if c['confirmed_missing']+t['confirmed_missing'] else ('TRANSPORT_UNKNOWN' if c['unresolved']+t['unresolved'] else 'SOURCE_BYTES_PRESENT')
        results.append(dict(row,source_status=status,**{'ct_'+k:v for k,v in c.items()},**{'teacher_'+k:v for k,v in t.items()},
                            supervision_admission='NOT_RUN',training_admitted=False))
    csv_write(out/'ADDED_DEPENDENCIES.csv',results)
    added=[]
    for case in plan['cases']:
        rr=[r for r in results if r['segment']==case['segment']]
        added.append(dict(id=case['id'],candidates=len(rr),restricted=sum(int(r['in_restricted_A']) for r in rr),
            source_bytes_present=sum(r['source_status']=='SOURCE_BYTES_PRESENT' for r in rr),
            ct_complete=sum(r['ct_present']==r['ct_required'] for r in rr),
            teacher_complete=sum(r['teacher_present']==r['teacher_required'] for r in rr),
            unknown_source=sum(r['source_status']=='TRANSPORT_UNKNOWN' for r in rr),
            known_missing_source=sum(r['source_status']=='SOURCE_MISSING' for r in rr)))
    original=a.work/'original';op,of,oc=receipts(original)
    rr=list(csv.DictReader((original/'ANNOTATION_CANDIDATES.csv').open()))
    originals=[];native_summaries=[]
    @functools.lru_cache(maxsize=256)
    def chunk(winding,cy,cx):
        key=f'pherc0139-{winding}-ct-y{cy:04}-x{cx:04}';rec=of[key]
        assert rec['status']=='PRESENT'
        return np.fromfile(original/rec['request']['path'],np.uint8).reshape(28,128,128)
    for winding in ['w035','w039','w040','w041','w044']:
        with np.load(original/'decoded'/f'{winding}.npz',allow_pickle=False) as f:
            target=f['target'];mask=f['supervision']
        assert not np.any(target&~mask)
        selected=[]
        for row in rr:
            if row['winding']!=winding:continue
            x,y=int(row['x']),int(row['y']);row=dict(row)
            needed=[f'pherc0139-{winding}-ct-y{cy:04}-x{cx:04}' for cy in range(y//128,(y+127)//128+1) for cx in range(x//128,(x+127)//128+1)]
            counts=source_status(needed,of);row.update({'ct_'+k:v for k,v in counts.items()})
            if counts['confirmed_missing'] or counts['unresolved']:
                row.update(status='SOURCE_MISSING' if counts['confirmed_missing'] else 'TRANSPORT_UNKNOWN',
                    supervised_after_ct='',positive_after_ct='',negative_after_ct='',all21_nonzero_supervised='',technically_eligible=False)
            else:
                patch=np.empty((28,128,128),np.uint8)
                for cy in range(y//128,(y+127)//128+1):
                    for cx in range(x//128,(x+127)//128+1):
                        y0=max(y,cy*128);y1=min(y+128,(cy+1)*128);x0=max(x,cx*128);x1=min(x+128,(cx+1)*128)
                        patch[:,y0-y:y1-y,x0-x:x1-x]=chunk(winding,cy,cx)[:,y0-cy*128:y1-cy*128,x0-cx*128:x1-cx*128]
                m=mask[y:y+128,x:x+128]&(patch[14]>0);t=target[y:y+128,x:x+128]
                count=int(m.sum());pos=int((m&t).sum())
                row.update(status='CT_CHECKED',supervised_after_ct=count,positive_after_ct=pos,negative_after_ct=count-pos,
                    all21_nonzero_supervised=int((m&np.all(patch[4:25]>0,axis=0)).sum()),technically_eligible=count>=1024)
            selected.append(row);originals.append(row)
        supported=[r for r in selected if r['status']=='CT_CHECKED']
        native_summaries.append(dict(winding=winding,mask_candidates=len(selected),ct_checked=len(supported),
            transport_unknown=sum(r['status']=='TRANSPORT_UNKNOWN' for r in selected),
            confirmed_missing=sum(r['status']=='SOURCE_MISSING' for r in selected),
            technically_eligible=sum(r['technically_eligible'] for r in selected),
            supervised_pixels_with_patch_overlap=sum(r['supervised_after_ct'] for r in supported),
            positive_pixels_with_patch_overlap=sum(r['positive_after_ct'] for r in supported),
            zero_positive_patches=sum(r['positive_after_ct']==0 for r in supported),
            supervision_removed_by_ct=sum(int(r['supervised_before_ct'])-r['supervised_after_ct'] for r in supported)))
    csv_write(out/'ORIGINAL_SUPPORT.csv',originals)
    outcome=dict(status='PARTIAL_INPUT_QUALIFICATION_NOT_TRAINING_ADMISSION',utc=stamp(),
        implementation_sha256=sha(Path(__file__)),added_request_counts=source_counts,added_cases=added,
        original_request_counts=oc,original_cases=native_summaries,
        new_model_forwards=0,paid_compute=False,full_cohort_admitted=False,
        added_teacher_population_class_counts='Unavailable; no availability-selected subset is promoted to the intended cohort',
        unknown_inputs_are_not_negatives=True,source_missing_is_distinct_from_transport_unresolved=True,
        output_pins={p.name:sha(p) for p in out.glob('*.csv')})
    write(out/'RESULTS.json',outcome)
    print(json.dumps(outcome,indent=2),flush=True)


if __name__=='__main__':main()
