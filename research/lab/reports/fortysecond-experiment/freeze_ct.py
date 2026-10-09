import ast
from datetime import timedelta
import numpy as np
from common import *
fixture=json.loads((HERE/'ACQUISITION_TESTS.json').read_text());need(fixture['tests_run']==fixture['passed']==3,'Acquisition fixtures')
selection=json.loads((HERE/'geometry/SELECTION.json').read_text());meta=json.loads((PRIOR/'PAIR_FEASIBILITY.json').read_text());chunks={};blocks=[]
bases=dict(old='https://data.aws.ash2txt.org/samples/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr/0/',new='https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0/')
for c in selection['candidates']:
 for volume,key in [('old','old_xyz'),('new','new_xyz_published_transform')]:
  center=np.asarray(c[key])[::-1];start=(np.rint(center/128).astype(int)-1)*128;need((start>=0).all() and (start+256<=meta[volume+'_shape_ZYX']).all(),'Native block in volume')
  ids=[]
  for dz in range(2):
   for dy in range(2):
    for dx in range(2):
     ix=(start//128+[dz,dy,dx]).tolist();identity=volume+'-'+'-'.join(map(str,ix));chunks[identity]=dict(id=identity,volume=volume,index_zyx=ix,url=bases[volume]+'/'.join(map(str,ix)));ids.append(identity)
  blocks.append(dict(id=c['id']+'-'+volume,pair=c['id'],nominal_role=c['nominal_role'],volume=volume,start_zyx=start.tolist(),center_zyx=center.tolist(),shape=[256]*3,chunk_ids=ids))
overlaps=[]
for i,a in enumerate(blocks):
 for b in blocks[i+1:]:
  if a['volume']==b['volume'] and np.all(np.maximum(a['start_zyx'],b['start_zyx'])<np.minimum(np.array(a['start_zyx'])+256,np.array(b['start_zyx'])+256)):overlaps.append([a['id'],b['id']])
need(not overlaps,'All full block footprints disjoint');need(len(chunks)<=96,'At most192MiB native CT')
first=json.loads((HERE/'GEOMETRY_GO.json').read_text())
plan=dict(status='FROZEN_BEFORE_CT',created_at_utc=now(),selection=pin(HERE/'geometry/SELECTION.json'),blocks=blocks,chunks=[chunks[k] for k in sorted(chunks)],planned_decoded_CT_bytes=len(chunks)*2097152,hard_CT_cap_bytes=256*1024**2,full_same_volume_block_overlaps=overlaps,admission_deadline_utc=(datetime.fromisoformat(first['created_at_utc'])+timedelta(minutes=45)).isoformat(),new_model_calls_allowed=0)
save(HERE/'CT_PLAN.json',plan)
paths=[HERE/x for x in ['PROTOCOL.md','common.py','acquire_ct.py','test_acquisition.py','ACQUISITION_TESTS.json','CT_PLAN.json','freeze_ct.py']]+[HERE/'geometry/SELECTION.json',PRIOR/'PAIR_FEASIBILITY.json']
for path in paths:
 if path.suffix=='.py':ast.parse(path.read_text())
save(HERE/'CT_GO.json',dict(status='GO_EXACT_NATIVE_CHUNKS_ONLY',created_at_utc=now(),pins=[pin(p) for p in paths],model_calls_allowed=0,chunk_count=len(chunks)))
print(json.dumps(dict(status='GO',GO=pin(HERE/'CT_GO.json'),chunks=len(chunks),bytes=plan['planned_decoded_CT_bytes'],deadline=plan['admission_deadline_utc'])))
