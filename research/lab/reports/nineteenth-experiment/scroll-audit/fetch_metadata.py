"""Bounded public metadata fetches only; no redirects or automatic retries."""
import datetime, hashlib, json, pathlib, sys, urllib.error, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
LEDGER = ROOT / 'request-ledger.json'
MAX_REQUESTS = 24
MAX_BYTES = 16 * 1024 * 1024

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def main():
    plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'max_requests':MAX_REQUESTS,'max_body_bytes':MAX_BYTES,'requests':[]}
    opener = urllib.request.build_opener(NoRedirect())
    (ROOT/'sources').mkdir(exist_ok=True)
    for item in plan:
        assert len(ledger['requests']) < MAX_REQUESTS
        assert item['url'].startswith(('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/','https://api.github.com/repos/ScrollPrize/villa/','https://raw.githubusercontent.com/ScrollPrize/villa/'))
        used = sum(x['body_bytes'] for x in ledger['requests'])
        remaining = MAX_BYTES-used
        assert remaining > 0
        target = ROOT/'sources'/item['name']
        assert not target.exists()
        rec = {'url':item['url'],'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':None,'error':None,'body_bytes':0,'body_path':str(target),'automatic_retries':0,'redirects_followed':0}
        body = b''
        try:
            try:
                response = opener.open(urllib.request.Request(item['url'],headers={'User-Agent':'Herculaneum-provenance-audit','Accept-Encoding':'identity'}), timeout=40)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                rec['status']=response.status
                rec['headers']={k.lower():v for k,v in response.headers.items() if k.lower() in {'content-length','content-type','last-modified','etag','date','location'}}
                chunks=[]
                while sum(map(len,chunks)) < remaining:
                    chunk=response.read(min(65536,remaining-sum(map(len,chunks))))
                    if not chunk: break
                    chunks.append(chunk)
                body=b''.join(chunks)
                rec['budget_boundary_reached']=len(body)==remaining
        except Exception as error:
            rec['error']=type(error).__name__+': '+str(error)
        target.write_bytes(body)
        rec.update(body_bytes=len(body),body_sha256=hashlib.sha256(body).hexdigest(),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        ledger['requests'].append(rec)
        ledger['body_bytes']=sum(x['body_bytes'] for x in ledger['requests'])
        LEDGER.write_text(json.dumps(ledger,indent=2)+'\n')
        print(item['name'],rec['status'],len(body),rec['error'])
        if rec.get('budget_boundary_reached'): break

if __name__ == '__main__': main()
