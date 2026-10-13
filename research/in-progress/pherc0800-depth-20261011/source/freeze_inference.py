"""Seal the passed qualifications and execution identities before production."""
import json,datetime
from common import *
go=json.loads((P/'INPUT_GO.json').read_text());q=json.loads((P/'BATCH_QUALIFICATION.json').read_text());assert q['status']=='PASS'
assert len(q['cases'])==32 and len(q['worker_baseline_replays'])==16 and q['new_forward_calls']==288 and q['tile_evaluations']==768
for r in go['pins']+go['runtime_pins']+[m['checkpoint'] for m in go['models']]:assert pin(r['path'])==r,r['path']
go['pins'] += [pin(P/'INPUT_GO.json'),pin(P/'BATCH_QUALIFICATION.json')]
go['status']='FROZEN_FOR_SHIFTED_PRODUCTION';go['created_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
go['estimated_model_seconds']=sum(sum(len(b['cohort_yx']) for b in g['batches'])*next(r['batch4_seconds']/4 for r in q['cases'] if (r['model'],r['depth'],r['phase'],r['direction'])==(m['name'],d,g['phase'],direction)) for m in go['models'] for d in [-15,15] for direction in ['forward','reverse'] for g in go['grids'] if g['family']==('hecate' if m['name']=='hecate' else 'villa'))
save(P/'INFERENCE_GO.json',go);print(json.dumps(dict(status=go['status'],go_sha256=pin(P/'INFERENCE_GO.json')['sha256'],expected_maps=32,expected_batches=3096,estimated_model_hours=go['estimated_model_seconds']/3600)))
