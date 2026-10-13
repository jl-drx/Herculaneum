"""Acquire hash-verified public Python sources and the fixed Hecate checkpoint."""
from pathlib import Path
import concurrent.futures, hashlib, json, requests, time
P=Path(__file__).resolve().parent
REV='0fb38c45b488e192566ea83ae36369cb6db92652'
tree=json.loads((P/'source/villa-ink-tree.json').read_text())
assert not tree['truncated']
items=[x for x in tree['tree'] if x['type']=='blob' and x['path'].startswith('vesuvius/src/vesuvius/') and x['path'].endswith(('.py','.json'))]
def fetch(item):
    url=f'https://raw.githubusercontent.com/ScrollPrize/villa/{REV}/{item["path"]}'
    r=requests.get(url,timeout=90);r.raise_for_status();data=r.content
    assert len(data)==item['size']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['sha']
    out=P/'runtime-source'/item['path'].removeprefix('vesuvius/src/')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    return dict(path=str(out),url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob=item['sha'])
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    records=list(pool.map(fetch,items))
(P/'RUNTIME_SOURCE_IDENTITY.json').write_text(json.dumps(dict(commit=REV,files=records),indent=2)+'\n')
print('source files verified',len(records),flush=True)
url='https://huggingface.co/scrollprize/hecate/resolve/9cb86e500e944b11a06a7020403cde5dffb5bcb2/hecate_9.6um.pth'
out=P/'work/hecate_9.6um.pth';sha=hashlib.sha256();n=0
with requests.get(url,stream=True,timeout=120) as r:
    r.raise_for_status()
    with out.with_suffix('.partial').open('xb') as f:
        for b in r.iter_content(4*1024*1024):f.write(b);sha.update(b);n+=len(b)
assert n==522576086
assert sha.hexdigest()=='809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d'
out.with_suffix('.partial').rename(out)
(P/'HECATE_IDENTITY.json').write_text(json.dumps(dict(url=url,path=str(out),bytes=n,sha256=sha.hexdigest()),indent=2)+'\n')
print('Hecate verified',n,flush=True)
