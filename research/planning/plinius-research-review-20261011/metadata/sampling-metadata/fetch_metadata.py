"""Bounded reads of public coordinate metadata; never follows redirects."""
import datetime, hashlib, json, urllib.request, urllib.error
from pathlib import Path
BASE=Path(__file__).resolve().parent
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

def main():
    plan=json.loads((BASE/'REQUEST_PLAN.json').read_text())
    out=BASE/'responses'; out.mkdir(exist_ok=True)
    ledger=[]; opener=urllib.request.build_opener(NoRedirect())
    for req in plan['requests']:
        item={'name':req['name'],'url':req['url'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        path=out/req['name']
        if path.exists(): raise RuntimeError('Refusing overwrite or silent reuse: '+str(path))
        try:
            with opener.open(urllib.request.Request(req['url'],headers={'User-Agent':'Plinius-public-metadata-research/1.0'}),timeout=20) as r:
                body=r.read(req['max_bytes']+1)
                if len(body)>req['max_bytes']: raise RuntimeError('Body limit exceeded')
                json.loads(body)
                path.write_bytes(body)
                item.update(status=r.status,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),etag=r.headers.get('ETag'))
        except urllib.error.HTTPError as e:
            item.update(status=e.code,error=type(e).__name__)
        except Exception as e:
            item.update(error=type(e).__name__,detail=str(e))
        ledger.append(item)
        (BASE/'REQUEST_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(req['name'],item.get('status',item.get('error')),item.get('bytes',0),flush=True)
    assert sum(i.get('bytes',0) for i in ledger)<=plan['total_body_cap']
if __name__=='__main__': main()
