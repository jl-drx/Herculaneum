"""Inspect a portable partial cache, or resume after a successful S3 probe."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import shutil
import urllib.request
import intake


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inspect','acquire'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--attempt');a=p.parse_args()
    plan=json.loads((a.work/'REQUEST_PLAN.json').read_text());pending=[];present=missing=0
    assert intake.sha(Path(intake.__file__))==plan['implementation_sha256']
    assert intake.sha(Path(__file__).with_name('PROTOCOL.md'))==plan['protocol_sha256']
    for rel,digest in plan['base_pins'].items():assert intake.sha(a.base/rel)==digest
    for r in plan['requests']:
        receipt=a.work/'receipts'/(r['id']+'.json')
        if not receipt.exists():pending.append(r);continue
        saved=json.loads(receipt.read_text());assert saved['request']==r
        if saved['status']=='PRESENT':
            source=a.work/r['path'];assert source.stat().st_size==r['bytes'] and intake.sha(source)==saved['sha256'];present+=1
        else:assert saved['status']=='MISSING';missing+=1
    summary=dict(status='LOCAL_CACHE_VERIFIED',present=present,confirmed_missing=missing,pending=len(pending),
        pending_body_bytes=sum(r['bytes'] for r in pending),request_plan_sha256=intake.sha(a.work/'REQUEST_PLAN.json'),
        network_requests=0,paid_compute=False)
    print(json.dumps(summary),flush=True)
    if a.mode=='inspect' or not pending:return
    if not a.attempt or not re.fullmatch('[a-zA-Z0-9_-]{1,60}',a.attempt):
        raise ValueError('Acquisition needs a new simple --attempt identifier')
    history=a.work/'continuations'/a.attempt;history.mkdir(parents=True,exist_ok=False)
    for name in ['ACQUISITION_STATUS.json','ACQUISITION_FAILURE.json','ACQUISITION_COMPLETE.json']:
        if (a.work/name).exists():shutil.copyfile(a.work/name,history/name)
    intake.write(history/'PLAN.json',dict(summary,utc=intake.stamp(),workers=4,
        implementation_sha256=intake.sha(Path(__file__)),eligibility_or_requests_changed=False,
        probe='One-byte range on first pending frozen object. A failed probe stops before bulk requests. A successful probe does not guarantee later requests.'))
    first=pending[0];headers={'Range':'bytes=0-0','Accept-Encoding':'identity'}
    if 'etag' in first:headers['If-Match']=first['etag']
    with urllib.request.urlopen(urllib.request.Request(first['url'],headers=headers),timeout=30) as r:
        length=first['bytes'] if first['kind']=='ct' else first['object_bytes']
        assert r.status==206 and r.headers['Content-Range']==f'bytes 0-0/{length}'
        if 'etag' in first:assert r.headers['ETag']==first['etag']
        assert len(r.read(2))==1
    intake.write(history/'PROBE.json',dict(status='PASS',utc=intake.stamp(),request_id=first['id'],body_bytes=1))
    executor=concurrent.futures.ThreadPoolExecutor
    def bounded_executor(*,max_workers):
        assert max_workers==16
        return executor(max_workers=4)
    concurrent.futures.ThreadPoolExecutor=bounded_executor
    intake.acquire(a)
    intake.write(history/'COMPLETE.json',dict(status='PASS',utc=intake.stamp(),
        acquisition_complete_sha256=intake.sha(a.work/'ACQUISITION_COMPLETE.json')))


if __name__=='__main__':main()
