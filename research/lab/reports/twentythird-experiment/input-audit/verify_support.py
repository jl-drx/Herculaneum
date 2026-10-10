"""Focused check of the retained primary support formula; no model or network."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
p=ROOT/'INPUT_CONTRACT.json';assert pin(p)['sha256']=='edb87179a3c8949bd43306f2c6e388c4de657ee9f52ea40f5e1192895c0ceafa';c=json.loads(p.read_text());rows=[]
border=np.zeros((384,384),dtype=np.bool_);border[32:352,32:352]=True
for case in c['cases']:
 a={}
 for key in ['primary_mask','known_mask','surface_mask']:
  record=case['arrays'][key];actual=pin(record['path']);assert all(actual[k]==record[k] for k in actual);a[key]=np.load(record['path'],allow_pickle=False)
 assert np.array_equal(a['primary_mask'],a['known_mask']&border)
 assert not np.any(a['known_mask']&~a['surface_mask'])
 rows.append({'id':case['id'],'primary_equals_known_and_central320_square':True,'known_subset_of_surface':True,'primary_pixels':int(a['primary_mask'].sum())})
source=ROOT.parents[1]/'fifteenth-experiment';records=[pin(source/'BENCHMARK_DESIGN.md'),pin(source/'implementation/select_labels.py')]
v={'status':'PASS','input_contract':pin(p),'code':pin(Path(__file__)),'source_contracts':records,'formula':'primary = known & central square [32:352,32:352] on384x384; known is a subset of supplied surface','known_origin':'Inherited phase15 source declaration: symmetric33x33 surface-inside and label-homogeneity native support, sampled at3x3 block centers; not recomputed from full native canvases here.','checks':rows,'model_calls':0,'network_calls':0};(ROOT/'SUPPORT_CONTRACT.json').write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','contexts':len(rows)}))
