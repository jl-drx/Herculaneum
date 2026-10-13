"""Frozen fine-coordinate GETs only; no CT/label/prediction objects."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,os
import requests
R=Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def one(item):
    p=R/'coordinates'/(item['id']+'.tif');rp=p.with_suffix('.json')
    assert sha(R/'intake/receipts'/(item['id']+'.json'))==item['head_receipt_sha256']
    if rp.exists():
        r=json.loads(rp.read_text());assert r['url']==item['url'] and r['etag']==item['etag']
        assert p.stat().st_size==item['bytes']==r['bytes'] and sha(p)==r['sha256'];return r
    assert not p.exists(),'Unreceipted coordinate file'
    tmp=p.with_suffix('.partial')
    with requests.get(item['url'],headers={'Accept-Encoding':'identity','If-Match':item['etag']},
                      stream=True,timeout=(15,60),allow_redirects=False) as x:
        assert x.status_code==200 and int(x.headers['Content-Length'])==item['bytes']
        assert x.headers.get('ETag')==item['etag'] and x.headers.get('Content-Encoding','identity')=='identity'
        n=0
        with tmp.open('xb') as f:
            for b in x.iter_content(65536):
                n+=len(b);assert n<=item['bytes'];f.write(b)
        assert n==item['bytes']
    os.replace(tmp,p);r=dict(item,sha256=sha(p),utc=datetime.now(timezone.utc).isoformat());save(rp,r)
    print(item['id'],n,flush=True);return r
def main():
    plan=json.loads((R/'intake/NEXT_FINE_GEOMETRY_PLAN.json').read_text())
    (R/'coordinates').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,plan['requests']))
    assert len(rows)==12 and sum(r['bytes'] for r in rows)==164736024==plan['max_body_bytes']
    save(R/'ACQUISITION.json',dict(status='COMPLETE',requests=rows,coordinate_bytes=sum(r['bytes'] for r in rows),
         CT_annotation_teacher_bytes=0,protocol_sha256=sha(R/'PROTOCOL.md'),code_sha256=sha(Path(__file__)),plan_sha256=sha(R/'intake/NEXT_FINE_GEOMETRY_PLAN.json')))
if __name__=='__main__':main()
