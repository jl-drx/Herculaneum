import datetime,hashlib,json,urllib.request,urllib.error
from pathlib import Path
b=Path(__file__).resolve().parent
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

def main():
    plan=json.loads((b/'REQUEST_PLAN.json').read_text()); out=b/'responses'; out.mkdir(exist_ok=True)
    ledger=[]; opener=urllib.request.build_opener(NoRedirect())
    for q in plan['requests']:
        item={'name':q['name'],'url':q['url'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        path=out/q['name']
        if path.exists(): raise RuntimeError('Refusing overwrite '+str(path))
        try:
            with opener.open(urllib.request.Request(q['url'],headers={'User-Agent':'Plinius-bounded-geometry/1.0'}),timeout=30) as r:
                data=r.read(q['max_bytes']+1)
                assert len(data)<=q['max_bytes'], 'Body cap'
                if 'exact_bytes' in q: assert len(data)==q['exact_bytes'], 'Length mismatch'
                with path.open('xb') as f: f.write(data)
                item.update(status=r.status,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),etag=r.headers.get('ETag'))
        except urllib.error.HTTPError as e: item.update(status=e.code,error=type(e).__name__)
        except Exception as e: item.update(error=type(e).__name__,detail=str(e))
        ledger.append(item); (b/'REQUEST_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(q['name'],item.get('status',item.get('error')),item.get('bytes',0),flush=True)
    assert sum(r.get('bytes',0) for r in ledger)<=plan['total_body_cap']
if __name__=='__main__': main()
