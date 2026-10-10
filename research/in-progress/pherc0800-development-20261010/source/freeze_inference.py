"""Bind successful qualification and all input identities before the long worker."""
import json,datetime
from common import *
go=json.loads((P/'INPUT_GO.json').read_text());q=json.loads((P/'BATCH_QUALIFICATION.json').read_text());assert q['status']=='PASS'
for r in go['pins']+go['runtime_pins']+[r['checkpoint'] for r in go['models']]:assert pin(r['path'])==r,r['path']
go['pins'] += [pin(P/'INPUT_GO.json'),pin(P/'BATCH_QUALIFICATION.json')]
go['status']='FROZEN_FOR_LONG_INFERENCE';go['created_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
go['estimated_seconds_by_job']=[]
for model in go['models']:
 family='hecate' if model['name']=='hecate' else 'villa'
 for phase in ['plain','padded']:
  grid=next(g for g in go['grids'] if g['family']==family and g['phase']==phase)
  for direction in ['forward','reverse']:
   timing=next(r for r in q['models_and_orders'] if r['model']==model['name'] and r['direction']==direction)['batch4_seconds']/4
   tiles=sum(map(len,grid['cohorts_yx']));go['estimated_seconds_by_job'].append(dict(model=model['name'],phase=phase,direction=direction,tiles=tiles,batches=len(grid['cohorts_yx']),estimated_seconds=tiles*timing))
go['estimated_model_seconds']=sum(r['estimated_seconds'] for r in go['estimated_seconds_by_job'])
save(P/'INFERENCE_GO.json',go);print(json.dumps(dict(status=go['status'],jobs=len(go['estimated_seconds_by_job']),model_hours=go['estimated_model_seconds']/3600,total_batches=sum(r['batches'] for r in go['estimated_seconds_by_job']),total_tiles=sum(r['tiles'] for r in go['estimated_seconds_by_job']))))
