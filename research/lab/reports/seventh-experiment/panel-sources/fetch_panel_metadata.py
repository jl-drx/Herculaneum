"""Bounded public documentation/metadata reads; no automatic redirects."""
from datetime import datetime, timezone
import hashlib, json, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / 'metadata-ledger.json'
CAP = 8 * 1024 * 1024
MAX_REQUESTS = 80

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def clean(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.hostname or '', p.path, '', ''))

def allowed(url):
    p = urllib.parse.urlsplit(url)
    h = p.hostname or ''
    if p.scheme != 'https' or p.port not in (None,443) or p.username or p.password or not (
        h == 'huggingface.co' or h.endswith('.hf.co') or h.endswith('.huggingface.co') or
        h == 'vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com' or h == 'raw.githubusercontent.com'):
        raise ValueError('Outside permitted public HTTPS sources')

def save(ledger):
    tmp = LEDGER.with_suffix('.partial')
    tmp.write_text(json.dumps(ledger, indent=2)+'\n')
    tmp.replace(LEDGER)

def fetch(url, name, object_cap=2*1024*1024):
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        'response_body_cap_bytes':CAP, 'maximum_http_requests':MAX_REQUESTS,
        'body_bytes':0, 'requests':[], 'artifacts':[]}
    if any(a['name']==name for a in ledger['artifacts']):
        raise ValueError('Refuse duplicate named metadata acquisition')
    stable_url=url
    opener=urllib.request.build_opener(NoRedirect())
    for hop in range(6):
        allowed(url)
        if len(ledger['requests'])>=MAX_REQUESTS or ledger['body_bytes']>=CAP:
            raise ValueError('Shared metadata budget exhausted')
        entry={'name':name,'url_without_query':clean(url),'hop':hop,
               'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'status':None,'body_bytes':0}
        ledger['requests'].append(entry)
        save(ledger)
        parts=[]
        try:
            try:
                response=opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=45)
            except urllib.error.HTTPError as exc:
                response=exc
            with response:
                status=response.status
                entry['status']=status
                headers={k.lower():v for k,v in response.headers.items()}
                entry['headers']={k:v for k,v in headers.items() if k in {
                    'content-length','content-encoding','content-type','etag','last-modified','date'}}
                if headers.get('content-encoding','identity').lower() not in ('','identity'):
                    raise ValueError('Unexpected transport encoding')
                limit=min(object_cap if status==200 else 4096,CAP-ledger['body_bytes'])
                declared=headers.get('content-length')
                if declared is not None and (int(declared)<0 or int(declared)>limit):
                    raise ValueError('Declared body exceeds budget; body unread')
                expected=int(declared) if declared is not None else None
                while entry['body_bytes']<limit:
                    n=min(8192,limit-entry['body_bytes'])
                    if expected is not None:
                        n=min(n,expected-entry['body_bytes'])
                        if n==0:
                            break
                    block=response.read(n)
                    if not block:
                        break
                    parts.append(block)
                    entry['body_bytes']+=len(block)
                    ledger['body_bytes']+=len(block)
                if expected is not None and entry['body_bytes']!=expected:
                    raise ValueError('Incomplete body')
                if expected is None and entry['body_bytes']==limit:
                    raise ValueError('Undeclared body reached bounded limit')
                body=b''.join(parts)
                entry['body_sha256']=hashlib.sha256(body).hexdigest()
                if status in (301,302,303,307,308):
                    location=response.headers.get('Location')
                    if not location or hop==5:
                        raise ValueError('Redirect contract failed')
                    url=urllib.parse.urljoin(url,location)
                    allowed(url)
                    entry['redirect_target_without_query']=clean(url)
                    save(ledger)
                    continue
                if status!=200:
                    raise ValueError('Metadata returned non-200')
                path=ROOT/('source-'+entry['body_sha256']+'-'+name)
                if path.exists():
                    raise ValueError('Refuse overwrite of existing metadata file')
                path.write_bytes(body)
                artifact={'name':name,'url':stable_url,'path':str(path),'sha256':entry['body_sha256'],
                          'byte_length':len(body),'etag':headers.get('etag')}
                ledger['artifacts'].append(artifact)
                save(ledger)
                return body,headers,artifact
        except Exception as exc:
            entry['error_type']=type(exc).__name__
            entry['body_sha256']=hashlib.sha256(b''.join(parts)).hexdigest()
            save(ledger)
            raise RuntimeError('Bounded metadata read failed ('+type(exc).__name__+')') from None
    raise RuntimeError('Redirect limit exceeded')

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('url');p.add_argument('name')
    a=p.parse_args();body,headers,record=fetch(a.url,a.name)
    print(json.dumps({'name':record['name'],'sha256':record['sha256'],'byte_length':len(body)}))
