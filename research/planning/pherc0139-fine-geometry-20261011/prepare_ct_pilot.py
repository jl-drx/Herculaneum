"""Freeze bounded CT requests using metadata and HEAD only. Fetch no CT body."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import argparse,json
import requests
import geometry as G
R=Path(__file__).resolve().parent;I=R/'intake'
BASE='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'

def metadata(item):
    p=R/'pilot-metadata'/(item['id']+'.json');rp=R/'pilot-receipts'/(item['id']+'.json')
    if rp.exists():
        old=G.read(rp);assert old['url']==item['url'] and G.sha(p)==old['sha256'];return old
    with requests.get(item['url'],stream=True,headers={'Accept-Encoding':'identity'},timeout=(15,45),allow_redirects=False) as x:
        assert x.status_code==200
        body=b''
        for block in x.iter_content(8192):
            body+=block;assert len(body)<=65536
        json.loads(body);p.write_bytes(body)
        row=dict(item,status=200,etag=x.headers.get('ETag'),bytes=len(body),sha256=G.sha(p),utc=datetime.now(timezone.utc).isoformat())
    G.save(rp,row);return row

def head(item):
    rp=R/'pilot-receipts'/(item['id']+'.json')
    if rp.exists():
        old=G.read(rp);assert old['url']==item['url'];return old
    with requests.head(item['url'],headers={'Accept-Encoding':'identity'},timeout=(15,45),allow_redirects=False) as x:
        length=int(x.headers.get('Content-Length',0))
        row=dict(item,status=x.status_code,content_length=length,etag=x.headers.get('ETag'),body_bytes=0,
                 usable=x.status_code==200 and length==item['expected_bytes'] and bool(x.headers.get('ETag')),
                 utc=datetime.now(timezone.utc).isoformat())
    G.save(rp,row);return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    cat=G.read(I/'CATALOG_EXTRACT.json');result=G.read(R/'qualification/RESULTS.json')
    selected=[r for r in result['regions'] if r.get('pilot',{}).get('geometry_ready')]
    assert len(selected)==8
    for directory in ['pilot-metadata','pilot-receipts']:(R/directory).mkdir(exist_ok=True)
    asks=[]
    for w in ['w040','w041','w043']:
        path=G.asset(cat,w,'layers-zarr','20260102150214')['origins'][0]['path']
        asks.append(dict(id=w+'-fine78-group2-array',method='GET',url=BASE+path+'2/.zarray',max_bytes=65536))
    plan=dict(requests=asks,max_body_bytes=3*65536,CT_pixel_bodies=0,qualification_sha256=G.sha(R/'qualification/RESULTS.json'))
    pp=R/'PILOT_METADATA_PLAN.json'
    if pp.exists():assert G.read(pp)==plan
    else:G.save(pp,plan)
    if a.prepare:print('Metadata plan frozen');return
    with ThreadPoolExecutor(max_workers=2) as pool:meta=list(pool.map(metadata,asks))
    requests_by_id={};sites=[]
    for r in selected:
        w=r['winding'];ids=[]
        for kind,volume,level,rect in [
            ('native','20250728140407',0,[r['native_context_xywh'][0],r['native_context_xywh'][1],r['native_context_xywh'][0]+256,r['native_context_xywh'][1]+256]),
            ('fine78','20260102150214',2,r['pilot']['fine_group2_xyxy'])]:
            mp=I/'metadata'/f'{w}-native-array.json' if kind=='native' else R/'pilot-metadata'/f'{w}-fine78-group2-array.json'
            arr=G.read(mp);depth=28 if kind=='native' else 109
            assert arr['shape'][0]==depth and arr['chunks']==[depth,128,128] and arr['dtype']=='|u1' and arr['compressor'] is None
            assert arr.get('filters') is None and arr['order']=='C'
            x0,y0,x1,y1=rect;assert 0<=x0<x1<=arr['shape'][2] and 0<=y0<y1<=arr['shape'][1]
            path=G.asset(cat,w,'layers-zarr',volume)['origins'][0]['path'];sep=arr.get('dimension_separator','.')
            for cy in range(y0//128,(y1-1)//128+1):
                for cx in range(x0//128,(x1-1)//128+1):
                    key=sep.join(map(str,(0,cy,cx)));id=f'{w}-{kind}-L{level}-0-{cy}-{cx}';ids.append(id)
                    request=dict(id=id,winding=w,kind=kind,level=level,key=key,url=BASE+path+str(level)+'/'+key,
                        expected_bytes=depth*128*128,array_metadata_path=str(mp.relative_to(R)),array_metadata_sha256=G.sha(mp))
                    if id in requests_by_id:assert requests_by_id[id]==request
                    else:requests_by_id[id]=request
        sites.append(dict(core_id=r['core_id'],winding=w,selection_order=r['selection_order'],native_context_xywh=r['native_context_xywh'],
            scored_core_xywh=r['core_xywh'],fine_group2_xyxy=r['pilot']['fine_group2_xyxy'],request_ids=ids))
    requests_list=[requests_by_id[k] for k in sorted(requests_by_id)]
    head_plan=dict(method='HEAD',requests=requests_list,max_CT_body_bytes=0)
    hp=R/'PILOT_HEAD_PLAN.json'
    if hp.exists():assert G.read(hp)==head_plan
    else:G.save(hp,head_plan)
    with ThreadPoolExecutor(max_workers=2) as pool:heads=list(pool.map(head,requests_list))
    hmap={x['id']:x for x in heads}
    for row in requests_list:
        h=hmap[row['id']];row.update(etag=h['etag'],head_qualified=h['usable'],head_receipt_sha256=G.sha(R/'pilot-receipts'/(row['id']+'.json')),sha256_after_GET=None)
    for s in sites:s['all_chunk_heads_qualified']=all(hmap[k]['usable'] for k in s['request_ids'])
    output=dict(status='FROZEN_NOT_EXECUTED' if all(h['usable'] for h in heads) else 'BLOCKED_SOURCE_HEAD_FAILURE',
        purpose='CT-only local correspondence pilot; no annotation/teacher payloads or training',
        fixed_selected_sites=sites,failed_geometry_sites=[r['core_id'] for r in result['regions'] if 'pilot' in r and not r['pilot']['geometry_ready']],
        requests=requests_list,max_CT_payload_bytes=sum(x['expected_bytes'] for x in requests_list),
        per_kind={kind:dict(chunks=sum(x['kind']==kind for x in requests_list),bytes=sum(x['expected_bytes'] for x in requests_list if x['kind']==kind)) for kind in ['native','fine78']},
        actual_metadata_GET_bytes=sum(x['bytes'] for x in meta),actual_CT_bytes_fetched=0,max_workers=2,automatic_retries=0,
        qualification_sha256=G.sha(R/'qualification/RESULTS.json'),preconditions=['Freeze CT local comparison/scoring protocol before GET',
            'GET with If-Match, verify exact lengths, retain SHA256 receipts','Audit full-depth source/nonzero support before scoring',
            'No replacement for geometry/source/correspondence failures'],
        limitation='Only eight predefined sites; geometry does not establish pointwise registration or winding-wide paired supervision')
    G.save(R/'NEXT_CT_PILOT_PLAN.json',output)
    print(json.dumps({k:output[k] for k in ['status','max_CT_payload_bytes','per_kind','actual_metadata_GET_bytes','actual_CT_bytes_fetched']}),flush=True)

if __name__=='__main__':main()
