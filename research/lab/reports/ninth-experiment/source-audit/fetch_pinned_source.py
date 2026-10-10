"""At most three exact pinned Villa source files; counted manual HTTPS redirects."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.parse,urllib.request,urllib.error
ROOT=Path(__file__).resolve().parent
COMMIT='e0bbb8b40a2db58b1d71864f286eb85717e59e64'
CAP=128*1024
MAX_REQUESTS=18
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None
def save(d):
    p=ROOT/'source-request-ledger.json';q=p.with_suffix('.partial');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def permitted(url):
    p=urllib.parse.urlsplit(url)
    if p.scheme!='https' or p.hostname!='raw.githubusercontent.com' or p.port not in (None,443) or p.username or p.password or not p.path.startswith('/ScrollPrize/villa/'+COMMIT+'/'):
        raise ValueError('Outside exact pinned public HTTPS repository')
def clean(url):
    p=urllib.parse.urlsplit(url);return urllib.parse.urlunsplit((p.scheme,p.hostname,p.path,'',''))
def fetch(relative,name):
    p=ROOT/'source-request-ledger.json'
    d=json.loads(p.read_text())if p.exists()else{'cap_response_body_bytes':CAP,'maximum_http_requests':MAX_REQUESTS,'body_bytes':0,'requests':[],'artifacts':[]}
    if len(d['artifacts'])>=3 or any(a['path']==str(ROOT/name)for a in d['artifacts']):raise ValueError('Source file limit or duplicate')
    url='https://raw.githubusercontent.com/ScrollPrize/villa/'+COMMIT+'/'+relative
    origin=url;opener=urllib.request.build_opener(NoRedirect())
    for hop in range(6):
        permitted(url)
        if len(d['requests'])>=MAX_REQUESTS or d['body_bytes']>=CAP:raise ValueError('Shared budget exhausted')
        e={'url_without_query':clean(url),'hop':hop,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'status':None,'body_bytes':0};d['requests'].append(e);save(d)
        parts=[]
        try:
            try:response=opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=45)
            except urllib.error.HTTPError as exc:response=exc
            with response:
                e['status']=response.status
                h={k.lower():v for k,v in response.headers.items()}
                e['headers']={k:v for k,v in h.items()if k in ['content-length','content-type','content-encoding','etag','last-modified','date']}
                if h.get('content-encoding','identity').lower()not in ('','identity'):raise ValueError('Unexpected transport encoding')
                limit=CAP-d['body_bytes'] if response.status==200 else min(4096,CAP-d['body_bytes'])
                declared=int(h['content-length'])if'content-length'in h else None
                if declared is not None and not 0<=declared<=limit:raise ValueError('Declared body exceeds budget; unread')
                while e['body_bytes']<limit:
                    n=min(4096,limit-e['body_bytes'])
                    if declared is not None:
                        n=min(n,declared-e['body_bytes'])
                        if n==0:break
                    block=response.read(n)
                    if not block:break
                    parts.append(block);e['body_bytes']+=len(block);d['body_bytes']+=len(block)
                if declared is not None and e['body_bytes']!=declared:raise ValueError('Truncated response')
                if declared is None and e['body_bytes']==limit:raise ValueError('Undeclared response reached limit')
                body=b''.join(parts);e['body_sha256']=hashlib.sha256(body).hexdigest()
                if response.status in (301,302,303,307,308):
                    location=response.headers.get('Location')
                    if not location or hop==5:raise ValueError('Redirect limit or missing target')
                    url=urllib.parse.urljoin(url,location);permitted(url);e['redirect_target_without_query']=clean(url);save(d);continue
                if response.status!=200:raise ValueError('Non-200 source; no retry')
                target=ROOT/name
                if target.exists():raise ValueError('Refuse source overwrite')
                target.write_bytes(body)
                a={'url':origin,'path':str(target),'sha256':e['body_sha256'],'byte_length':len(body),'commit':COMMIT}
                d['artifacts'].append(a);save(d);return body,a
        except Exception as exc:
            e['error_type']=type(exc).__name__;e['body_sha256']=hashlib.sha256(b''.join(parts)).hexdigest();save(d)
            raise RuntimeError('Bounded source acquisition stopped ('+type(exc).__name__+')')from None
    raise RuntimeError('Redirect limit')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('relative');p.add_argument('name');a=p.parse_args()
    body,record=fetch(a.relative,a.name);print(json.dumps({'path':record['path'],'bytes':len(body),'sha256':record['sha256']}))
