"""Nine exact, one-shot public object reads; refuse overwrite, redirect or mismatch."""
import datetime, hashlib, json, urllib.request, urllib.error
from pathlib import Path
BASE=Path(__file__).resolve().parent
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def main():
    frozen=json.loads((BASE/'SOURCE_FREEZE.json').read_text())
    for f,h in frozen['sha256'].items(): assert hashlib.sha256((BASE/f).read_bytes()).hexdigest()==h, f
    plan=json.loads((BASE/'REQUEST_PLAN.json').read_text());ledger=[];out=BASE/'chunks';out.mkdir(exist_ok=False)
    opener=urllib.request.build_opener(NoRedirect())
    for req in plan['requests']:
        item={**req,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            with opener.open(urllib.request.Request(req['url'],headers={'User-Agent':'Plinius-CT-qualification/1.0'}),timeout=30) as r:
                body=r.read(req['expected_bytes']+1)
                item.update(status=r.status,bytes=len(body),etag=r.headers.get('ETag'),sha256=hashlib.sha256(body).hexdigest())
                if r.status!=200 or len(body)!=req['expected_bytes']:raise ValueError('Unexpected status/length')
                (out/req['name']).write_bytes(body)
        except Exception as e:
            item.update(error=type(e).__name__,detail=str(e));ledger.append(item)
            (BASE/'REQUEST_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n');raise
        ledger.append(item);(BASE/'REQUEST_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(req['name'],item['bytes'],item['sha256'],flush=True)
    assert sum(i['bytes'] for i in ledger)==plan['total_body_cap']
if __name__=='__main__':main()
