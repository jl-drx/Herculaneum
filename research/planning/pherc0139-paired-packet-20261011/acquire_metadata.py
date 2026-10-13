"""Finite public metadata acquisition. No CT, label or prediction pixels."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,value):
    tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

def one(item):
    p=ROOT/'metadata'/(item['id']+'.json')
    rp=ROOT/'receipts'/(item['id']+'.json')
    if rp.exists():
        row=json.loads(rp.read_text());assert row['url']==item['url'] and row['method']==item['method']
        if item['method']=='GET':assert sha(p)==row['sha256'] and p.stat().st_size==row['body_bytes']
        return row
    with requests.request(item['method'],item['url'],headers={'Accept-Encoding':'identity'},
                          stream=True,timeout=(15,45),allow_redirects=False) as response:
        assert response.status_code==200,(item['id'],response.status_code)
        assert response.headers.get('Content-Encoding','identity')=='identity'
        length=int(response.headers['Content-Length'])
        row=dict(item,status=response.status_code,content_length=length,etag=response.headers.get('ETag'),
                 last_modified=response.headers.get('Last-Modified'),retrieved_utc=datetime.now(timezone.utc).isoformat(),body_bytes=0)
        if item['method']=='GET':
            assert length<=item['max_bytes']
            body=b''
            for block in response.iter_content(8192):
                body+=block;assert len(body)<=item['max_bytes']
            assert len(body)==length
            json.loads(body)  # Metadata only, not an opaque payload.
            p.write_bytes(body);row.update(body_bytes=len(body),sha256=sha(p))
        save(rp,row)
        return row

def main():
    plan=json.loads((ROOT/'METADATA_PLAN.json').read_text())
    assert sha(ROOT/'CATALOG_EXTRACT.json')==plan['catalog_sha256']
    for name in ['metadata','receipts']:(ROOT/name).mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan['max_workers']) as pool:
        results=list(pool.map(one,plan['requests']))
    summary=dict(complete=True,code_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'METADATA_PLAN.json'),
                 requests=len(results),body_bytes=sum(r['body_bytes'] for r in results),responses=results,
                 CT_or_prediction_or_label_payloads_fetched=0)
    assert summary['body_bytes']<=plan['max_body_bytes']
    save(ROOT/'METADATA_LEDGER.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='responses'}),flush=True)

if __name__=='__main__':main()
