"""Enumerate source chunks from frozen geometry only; no pixel I/O."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent; ROOT=B.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
geo=ROOT/'geometry-review/pullback'; results=json.loads((geo/'RESULTS.json').read_text()); assert results['complete']
cat=json.loads((ROOT/'metadata/CATALOG_EXTRACT.json').read_text())
requests=[]; sites=[]
vols={}
for key,vol,level in [('fine','20260102150214',2),('native','20250728140407',0)]:
    asset=next(a for a in cat['assets'] if a['type']=='layers-zarr' and a['parameters']['target_volume']==vol)
    path=asset['origins'][0]['path']
    meta=ROOT/f'metadata/responses/{vol}-layers-zarr-{level}-zarray.json'; z=json.loads(meta.read_text())
    assert z['compressor'] is None and z['filters'] is None and z['dtype']=='|u1' and z['order']=='C' and z['dimension_separator']=='/'
    vols[key]=dict(volume=vol,level=level,base_url='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'+path+str(level)+'/',array=z,metadata_sha256=sha(meta))
for site in results['sites']:
    sid=site['site']; sd=geo/sid
    for ent in site['output_files']: assert sha(sd/ent['name'])==ent['sha256']
    valid=np.load(sd/'support.npy'); assert valid.all(), 'Incomplete geometry; no silent replacement'
    q=np.load(sd/'queries_uvk.npy'); assert q.shape==(20,272,272,3)
    low=np.floor(q.min((0,1,2))).astype(int); high=np.floor(q.max((0,1,2))).astype(int)+1
    assert low[2]-5>=0 and high[2]+5<109,'Insufficient unpadded Gaussian depth context'
    x,y,w,h=site['native_xywh']; bounds={'fine':[int(low[0]),int(low[1]),int(high[0])+1,int(high[1])+1],'native':[x,y,x+w,y+h]}
    info=dict(site=sid,source_xy_half_open=bounds,native_xywh=site['native_xywh'],queries_sha256=sha(sd/'queries_uvk.npy'),support_sha256=sha(sd/'support.npy'),fine_depth_filter_footprint_inclusive=[int(low[2])-5,int(high[2])+5],
              local_geometry_result_sha256=sha(sd/'RESULTS.json'))
    sites.append(info)
    for key in ['fine','native']:
        a,b,c,d=bounds[key];v=vols[key];cz,cy,cx=v['array']['chunks']
        assert 0<=a<c<=v['array']['shape'][2] and 0<=b<d<=v['array']['shape'][1]
        for yy in range(b//cy,(d-1)//cy+1):
            for xx in range(a//cx,(c-1)//cx+1):
                name=f'{sid}-{key}-0-{yy}-{xx}.bin'
                requests.append(dict(name=name,site=sid,acquisition=key,chunk_zyx=[0,yy,xx],url=v['base_url']+f'0/{yy}/{xx}',exact_bytes=cz*cy*cx))
inputs=['PROTOCOL.md','structural_scores.py','FIXTURES.json','prepare_requests.py','../geometry-review/pullback/PLAN.json','../geometry-review/pullback/RESULTS.json','../geometry-review/pullback/LOCAL_BASES.json']
plan=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='First new CT reads after geometry-only selection and synthetic score checks',frozen_files=[dict(path=p,sha256=sha(B/p)) for p in inputs],volumes=vols,sites=sites,requests=requests,total_expected_bytes=sum(r['exact_bytes'] for r in requests),body_cap_bytes=192*1024**2,retries=0,redirects=False,missing_chunks='Stop; never substitute implicit zero fill',all_z_chunks_retained=True)
assert plan['total_expected_bytes']<=plan['body_cap_bytes']
with (B/'REQUEST_PLAN.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
print(json.dumps({k:plan[k] for k in ['frozen_utc','total_expected_bytes','body_cap_bytes']},indent=2));print('objects',len(requests))
