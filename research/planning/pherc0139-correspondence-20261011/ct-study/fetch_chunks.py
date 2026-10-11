"""Bounded exact public surface chunks; TLS/proxy inherited, no redirects/retries."""
import datetime,hashlib,json,urllib.request,urllib.error
from pathlib import Path
B=Path(__file__).resolve().parent
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**kw):return None
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((B/'REQUEST_PLAN.json').read_text())
    for f in plan['frozen_files']:assert sha(B/f['path'])==f['sha256'], 'Frozen input changed '+f['path']
    chunks=B/'chunks';chunks.mkdir(exist_ok=True); ledger=[];opener=urllib.request.build_opener(NoRedirect());total=0
    if (B/'REQUEST_LEDGER.json').exists():raise RuntimeError('Refusing overwrite/retry')
    for n,r in enumerate(plan['requests']):
        item=dict(name=r['name'],url=r['url'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            with opener.open(urllib.request.Request(r['url'],headers={'User-Agent':'Plinius-fixed-correspondence/1.0'}),timeout=40) as res:
                body=res.read(r['exact_bytes']+1);total+=len(body)
                item.update(status=res.status,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),etag=res.headers.get('ETag'))
                assert len(body)==r['exact_bytes'],'Unexpected chunk length'
                assert total<=plan['body_cap_bytes'],'Byte cap'
                with (chunks/r['name']).open('xb') as f:f.write(body)
        except Exception as e:
            item.update(error=type(e).__name__,detail=str(e));ledger.append(item)
            (B/'REQUEST_LEDGER.json').write_text(json.dumps(dict(plan_sha256=sha(B/'REQUEST_PLAN.json'),requests=ledger,bytes_read=total,complete=False),indent=2)+'\n');raise
        ledger.append(item)
        (B/'REQUEST_LEDGER.json').write_text(json.dumps(dict(plan_sha256=sha(B/'REQUEST_PLAN.json'),requests=ledger,bytes_read=total,complete=len(ledger)==len(plan['requests'])),indent=2)+'\n')
        if n%5==0 or n==len(plan['requests'])-1:print(n+1,'/',len(plan['requests']),'bytes',total,flush=True)
    assert total==plan['total_expected_bytes']
if __name__=='__main__':main()
