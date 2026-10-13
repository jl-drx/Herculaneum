"""Five fixed native annotation sources; supervision admission precedes labels."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import numpy as np
from intake import sha, stamp, write


def csv_write(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=False)
    lib=a.repo/'research/planning/pherc0139-learning-readiness-20261011/native-reference/intake.py'
    spec=importlib.util.spec_from_file_location('retained_native_annotation_decoder',lib)
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    cohort=json.loads((a.base/'COHORT_CANDIDATES.json').read_text())
    prior_plan=json.loads((a.prior/'SOURCE_PLAN.json').read_text())['arrays']
    prior_receipts=json.loads((a.prior/'REQUEST_LEDGER.json').read_text())['requests']
    ledger=[];items=[];used=0
    (a.work/'metadata').mkdir()
    def metadata_get(url,dest):
        nonlocal used
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=40) as r:
            assert r.status==200;data=r.read(65537);assert len(data)<=65536
            item=dict(url=url,bytes=len(data),etag=r.headers['ETag'],sha256=hashlib.sha256(data).hexdigest(),path=str(dest.relative_to(a.work)),utc=stamp())
        used+=len(data);assert used<4*1024**2
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);ledger.append(item)
        return json.loads(data)
    for source in cohort['original_stream']:
        winding=source['segment'].split('-')[1].split('_')[0]
        root='https://vesuvius-challenge-open-data.s3.amazonaws.com/'
        native=metadata_get(root+source['native_path'].rstrip('/')+'/0/.zarray',a.work/'metadata'/f'{winding}-native.json')
        assert native['shape'][0]==28 and native['chunks']==[28,128,128] and native['dtype']=='|u1' and native['compressor'] is None
        assert native['order']=='C' and native['dimension_separator']=='/' and native['filters'] is None
        for role in ['supervision','inklabels']:
            url=root+source['labels']['origins'][0]['path']+role+'.zarr/0/'
            dest=a.work/'arrays'/winding/role/'zarr.json'
            meta=metadata_get(url+'zarr.json',dest);decoder.layout(meta)
            assert meta['shape']==native['shape'][1:]
            shard_url=url+'c/0/0'
            with urllib.request.urlopen(urllib.request.Request(shard_url,method='HEAD'),timeout=40) as r:
                assert r.status==200;length=int(r.headers['Content-Length']);etag=r.headers['ETag']
            assert 0<length<=1024**2
            item=dict(winding=winding,role=role,segment=source['segment'],url=shard_url,
                bytes=length,etag=etag,metadata_sha256=sha(dest),path=f'arrays/{winding}/{role}/c/0/0',
                native_path=source['native_path'],native_shape=native['shape'])
            old=next((r for r in prior_plan if r['winding']==winding and r['role']==role),None)
            if old:
                rec=next(r for r in prior_receipts if r['winding']==winding and r['role']==role)
                assert length==old['content_length'] and etag==old['etag']
                assert sha(dest)==old['metadata_sha256']
                item['reuse']=dict(path=str(a.prior/rec['path']),sha256=rec['sha256'])
            items.append(item)
    source_plan=dict(utc=stamp(),items=items,minimum_supervised_pixels=1024,stride=96,patch_size=128,
        no_class_filter=True,cohort_sha256=sha(a.base/'COHORT_CANDIDATES.json'),
        protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),implementation_sha256=sha(Path(__file__)),
        decoder_sha256=sha(lib),population_label_payload_read=False)
    write(a.work/'ANNOTATION_SOURCE_PLAN.json',source_plan)
    def fetch(item):
        nonlocal used
        dest=a.work/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if 'reuse' in item:
            source=Path(item['reuse']['path']);assert sha(source)==item['reuse']['sha256']
            shutil.copyfile(source,dest);network_bytes=0
        else:
            with urllib.request.urlopen(urllib.request.Request(item['url'],headers={'If-Match':item['etag'],'Accept-Encoding':'identity'}),timeout=40) as r:
                assert r.status==200 and r.headers['ETag']==item['etag']
                data=r.read(item['bytes']+1);assert len(data)==item['bytes']
            dest.write_bytes(data);used+=len(data);assert used<4*1024**2;network_bytes=len(data)
        assert dest.stat().st_size==item['bytes']
        ledger.append(dict(item,sha256=sha(dest),network_bytes=network_bytes,utc=stamp()))
        meta=json.loads((dest.parents[2]/'zarr.json').read_text())
        decoded,checks=decoder.decode_crosscheck(dest.parents[2],meta)
        return decoded,checks
    masks={};summaries=[];rows=[];decoder_checks=[]
    for item in items:
        if item['role']!='supervision':continue
        winding=item['winding'];mask,checks=fetch(item);masks[winding]=mask!=0
        decoder_checks.append(dict(winding=winding,role='supervision',**checks))
        assert set(np.unique(mask))<={0,255}
        h,w=mask.shape;eligible=0
        for y in range(0,h-127,96):
            for x in range(0,w-127,96):
                count=int(masks[winding][y:y+128,x:x+128].sum())
                if count>=1024:
                    rows.append(dict(id=f'pherc0139-{winding}-x{x:05}-y{y:05}',winding=winding,segment=item['segment'],x=x,y=y,supervised_before_ct=count));eligible+=1
        summaries.append(dict(winding=winding,supervised_pixels=int(masks[winding].sum()),eligible=eligible))
    csv_write(a.work/'MASK_CANDIDATES.csv',rows)
    write(a.work/'MASK_FREEZE.json',dict(utc=stamp(),csv_sha256=sha(a.work/'MASK_CANDIDATES.csv'),
        source_plan_sha256=sha(a.work/'ANNOTATION_SOURCE_PLAN.json'),summaries=summaries,
        new_label_payloads_before_freeze=0,prior_exposure='Previously inspected native sources are exposed development data, not blind labels'))
    (a.work/'decoded').mkdir()
    for item in items:
        if item['role']!='inklabels':continue
        winding=item['winding'];labels,checks=fetch(item);mask=masks[winding]
        decoder_checks.append(dict(winding=winding,role='inklabels',**checks))
        assert set(np.unique(labels))<={0,255}
        np.savez_compressed(a.work/'decoded'/f'{winding}.npz',target=(labels>0)&mask,supervision=mask)
        for row in rows:
            if row['winding']!=winding:continue
            y,x=row['y'],row['x'];m=mask[y:y+128,x:x+128];t=labels[y:y+128,x:x+128]>0
            row.update(positive_before_ct=int((t&m).sum()),outside_mask_positive=int((t&~m).sum()))
        next(s for s in summaries if s['winding']==winding).update(
            ink_inside_supervision=int(((labels>0)&mask).sum()),ink_outside_supervision=int(((labels>0)&~mask).sum()))
    csv_write(a.work/'ANNOTATION_CANDIDATES.csv',rows)
    write(a.work/'ANNOTATION_QUALIFICATION.json',dict(status='MASKS_AND_LABELS_DECODED',utc=stamp(),
        candidates=len(rows),summaries=summaries,decode_checks=decoder_checks,new_body_bytes=used,
        input_ct_qualification_pending=True,model_forwards=0,paid_compute=False))
    write(a.work/'ANNOTATION_LEDGER.json',dict(requests=ledger,new_body_bytes=used))
    oldct=json.loads((a.prior/'ct/REQUEST_PLAN.json').read_text())['requests']
    previous={(r['winding'],r['chunk_zyx'][1],r['chunk_zyx'][2]):r for r in oldct}
    requests=[]
    for item in items:
        if item['role']!='supervision':continue
        winding=item['winding'];chunks=set()
        for r in rows:
            if r['winding']!=winding:continue
            x,y=r['x'],r['y']
            chunks.update((cy,cx) for cy in range(y//128,(y+127)//128+1) for cx in range(x//128,(x+127)//128+1))
        for cy,cx in sorted(chunks):
            r=dict(id=f'pherc0139-{winding}-ct-y{cy:04}-x{cx:04}',source_id=winding,kind='ct',cy=cy,cx=cx,
                url='https://vesuvius-challenge-open-data.s3.amazonaws.com/'+item['native_path'].rstrip('/')+f'/0/0/{cy}/{cx}',
                bytes=28*128*128,path=f'objects/{winding}/ct/{cy}/{cx}.bin')
            old=previous.get((winding,cy,cx))
            if old:
                receipt=json.loads((a.prior/'ct/receipts'/(old['id']+'.json')).read_text())
                r['reuse']=dict(path=str(a.prior/'ct'/old['relative_path']),sha256=receipt['sha256'])
                r['etag']=receipt['etag']
            requests.append(r)
    acquisition=Path(__file__).with_name('intake.py')
    write(a.work/'REQUEST_PLAN.json',dict(status='ORIGINAL_CT_FROZEN_AFTER_SUPERVISION_ADMISSION',utc=stamp(),
        requests=requests,planned_body_bytes=sum(r['bytes'] for r in requests),base_pins={},
        protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),implementation_sha256=sha(acquisition),
        maximum_concurrent_reads=16,maximum_body_bytes_with_retries=2*1024**3,minimum_supervised_pixels=1024,
        mask_freeze_sha256=sha(a.work/'MASK_FREEZE.json'),model_forwards=0,paid_compute=False))
    print(json.dumps(dict(candidates=len(rows),summaries=summaries,ct_chunks=len(requests),
                         new_ct_chunks=sum('reuse' not in r for r in requests))),flush=True)


if __name__=='__main__':main()
