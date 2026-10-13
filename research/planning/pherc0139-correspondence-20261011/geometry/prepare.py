import json,datetime
from pathlib import Path
b=Path(__file__).resolve().parent
inventories=json.loads((b.parent/'metadata/MESH_INVENTORIES.json').read_text())
requests=[]
for inv in inventories:
    for obj in inv['objects']:
        if not obj['key'].endswith('.tif'): continue
        volume='20260102150214' if '20260102150214' in obj['key'] else '20250728140407'
        requests.append(dict(name=volume+'-'+obj['key'].split('/')[-1],url='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'+obj['key'],max_bytes=obj['bytes'],exact_bytes=obj['bytes']))
for commit in ['650bd56663efb7225fdba73fe267ba86b8e29368','d9a472ea2f50d9ce4ce56303cc1d682d962490d9']:
    paths=['apps/src/vc_transform_geom.cpp']
    if commit.startswith('d9'): paths.append('core/src/QuadSurface.cpp')
    for path in paths:
        requests.append(dict(name=commit[:7]+'-'+path.split('/')[-1],url='https://raw.githubusercontent.com/ScrollPrize/villa/'+commit+'/volume-cartographer/'+path,max_bytes=524288))
p=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Coordinate payload and exact transformer implementation only; no CT pixels, labels or models.',requests=requests,total_body_cap=sum(r['max_bytes'] for r in requests))
with (b/'REQUEST_PLAN.json').open('x') as f: json.dump(p,f,indent=2); f.write('\n')
print(p['total_body_cap'],len(requests))
