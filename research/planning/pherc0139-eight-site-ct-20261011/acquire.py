"""Pinned finite CT GETs. Never retries or silently accepts changed objects."""
from common import *
from datetime import datetime,timezone
import requests
def main():
    check_lock();plan=read(PLAN);assert read(R/'PREFLIGHT_VERIFICATION.json')['status']=='PASS'
    assert read(R/'geometry/RESULTS.json')['ready']>0
    assert not (R/'ACQUISITION.json').exists();(R/'chunks').mkdir(exist_ok=False)
    session=requests.Session();rows=[];total=0;start=datetime.now(timezone.utc).isoformat()
    for req in plan['requests']:
        row=dict(id=req['id'],url=req['url'],started_utc=datetime.now(timezone.utc).isoformat())
        try:
            with session.get(req['url'],headers={'If-Match':req['etag'],'Accept-Encoding':'identity'},stream=True,timeout=(20,40),allow_redirects=False) as response:
                row.update(status=response.status_code,etag=response.headers.get('ETag'))
                assert response.status_code==200 and row['etag']==req['etag'],'Object identity/status changed'
                assert int(response.headers.get('Content-Length','-1'))==req['expected_bytes'],'Header size changed'
                body=response.raw.read(req['expected_bytes']+1,decode_content=False);total+=len(body)
                row.update(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
                assert len(body)==req['expected_bytes'] and total<=plan['max_CT_payload_bytes'],'Body/cap mismatch'
                (R/'chunks'/(req['id']+'.bin')).write_bytes(body)
        except Exception as exc:
            row.update(error=type(exc).__name__,detail=str(exc));rows.append(row)
            save(R/'ACQUISITION.json',dict(complete=False,started_utc=start,lock_sha256=sha(R/'LOCK.json'),plan_sha256=sha(PLAN),bytes_received=total,requests=rows));raise
        rows.append(row)
        save(R/'ACQUISITION.json',dict(complete=len(rows)==len(plan['requests']),started_utc=start,lock_sha256=sha(R/'LOCK.json'),plan_sha256=sha(PLAN),bytes_received=total,requests=rows))
        if len(rows)%8==0 or len(rows)==len(plan['requests']):print(len(rows),'/',len(plan['requests']),total,'bytes',flush=True)
    assert total==plan['max_CT_payload_bytes']==86114304
if __name__=='__main__':main()
