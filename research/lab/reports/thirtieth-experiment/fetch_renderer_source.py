"""Bounded version-specific public source inspection; never executes fetched code."""
import base64,hashlib,json,re,signal,ssl,sys,time,urllib.error,urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE/'renderer-source-v1'
BASE='https://api.github.com/repos/educelab/volume-cartographer/'


def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),size_bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def save(p,d):
    tmp=p.with_suffix(p.suffix+'.writing');tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');tmp.replace(p)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def main(relative,name):
    go=json.loads((HERE/'RENDERER_SOURCE_GO.json').read_text())
    for r in go['files']:assert pin(Path(r['path']))==r
    assert re.fullmatch(r'git/(?:matching-refs/tags/v?2\.26\.0|(?:tags|commits|blobs)/[0-9a-f]{40}|trees/[0-9a-f]{40}(?:\?recursive=1)?)',relative)
    assert re.fullmatch(r'[A-Za-z0-9_.-]+',name)
    ROOT.mkdir(exist_ok=True);dest=ROOT/(name+'.response.json');assert not dest.exists()
    ledger_path=ROOT/'LEDGER.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(requests=[],reserved_body_bytes=0,returned_body_bytes=0,retries=0,redirects=0)
    cap=1048576;assert len(ledger['requests'])<14
    entry=dict(url=BASE+relative,name=name,success=False,reserved_body_bytes=cap+1)
    ledger['requests'].append(entry);ledger['reserved_body_bytes']+=cap+1;save(ledger_path,ledger)
    body=b'';resp=None;start=time.monotonic()
    def expired(*_):raise TimeoutError('20second source request deadline')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,20)
    try:
        opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        try:resp=opener.open(urllib.request.Request(entry['url'],headers={'User-Agent':'Herculaneum-Renderer-Source-Audit','Accept-Encoding':'identity'}),timeout=15)
        except urllib.error.HTTPError as e:resp=e
        entry['status']=resp.status
        assert resp.headers.get('Content-Type','').split(';')[0] in ('application/json','application/vnd.github+json')
        assert resp.headers.get('Content-Encoding','identity')=='identity'
        body=resp.read(cap+1);assert len(body)<=cap and resp.status==200
        obj=json.loads(body)
        if relative.startswith('git/blobs/'):
            sha=relative.rsplit('/',1)[1];assert obj['sha']==sha and obj['encoding']=='base64'
            data=base64.b64decode(obj['content']);data.decode('utf-8')
            assert len(data)==obj['size'] and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==sha
            out=ROOT/(name+'.source');out.write_bytes(data);entry['source']=pin(out);entry['git_blob_sha1']=sha
        entry['success']=True
    except Exception as e:entry['error']=type(e).__name__+': '+str(e);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if resp:resp.close()
        dest.write_bytes(body);entry['response']=pin(dest);entry['elapsed_seconds']=time.monotonic()-start
        ledger['returned_body_bytes']+=len(body);save(ledger_path,ledger)
    print(json.dumps(entry,indent=2))


if __name__=='__main__':main(*sys.argv[1:])
