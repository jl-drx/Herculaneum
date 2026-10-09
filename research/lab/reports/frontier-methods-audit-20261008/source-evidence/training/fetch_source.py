"""Bounded, read-only public source acquisition; never weights or datasets."""
from pathlib import Path
from datetime import datetime, timezone
import urllib.request, urllib.error, hashlib, json
ROOT=Path(__file__).resolve().parent
CAP=7*1024*1024
JOURNAL=ROOT/'request-ledger.json'
def fetch(url,name,limit=262144,method='GET'):
    ledger=json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {'response_body_bytes':0,'requests':[],'cap_bytes':CAP}
    remaining=CAP-ledger['response_body_bytes']
    if remaining<=0:raise RuntimeError('Source-only budget exhausted')
    req=urllib.request.Request(url,headers={'User-Agent':'Herculaneum-frontier-methods-source-audit/1.0','Accept-Encoding':'identity'},method=method)
    rec={'url':url,'method':method,'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
    response=None
    try:
        try:response=urllib.request.urlopen(req,timeout=25)
        except urllib.error.HTTPError as exc:response=exc
        with response:
            rec.update(http_status=response.status,final_url=response.geturl(),etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'),content_encoding=response.headers.get('Content-Encoding'),declared_content_length=response.headers.get('Content-Length'))
            bound=min(limit,remaining)
            if rec['declared_content_length'] and int(rec['declared_content_length'])>bound:
                rec.update(response_body_bytes=0,body_skipped='Declared source size exceeds response bound')
                return None
            raw=response.read(bound) if method!='HEAD' else b''
            rec['response_body_bytes']=len(raw)
            if len(raw)==bound:rec['truncated_or_at_bound']=True
            if raw:
                digest=hashlib.sha256(raw).hexdigest();path=ROOT/f'{digest}-{name}'
                if path.exists():assert path.read_bytes()==raw
                else:path.write_bytes(raw);path.chmod(0o444)
                rec.update(sha256=digest,retained_path=str(path))
            return raw if response.status==200 and not rec.get('truncated_or_at_bound') else None
    except Exception as exc:
        rec.update(error_type=type(exc).__name__,response_body_bytes=rec.get('response_body_bytes',0))
        raise
    finally:
        ledger['response_body_bytes']+=rec.get('response_body_bytes',0);ledger['requests'].append(rec)
        JOURNAL.write_text(json.dumps(ledger,indent=2)+'\n')
