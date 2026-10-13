"""Bind the approved worker qualification without changing the frozen design."""
import json,datetime
from common import *
assert not (P/'INPUT_GO.json').exists()
prior=R/'fiftyfirst-experiment';old=json.loads((prior/'INFERENCE_GO.json').read_text())
assert pin(prior/'INFERENCE_GO.json')['sha256']=='bd10e35e367fa982a3db8bd14be581766df9e06b7629186e3dba54f318cfc05e'
for r in old['pins']+old['runtime_pins']+[m['checkpoint'] for m in old['models']]:assert pin(r['path'])==r
fg=json.loads((P/'FEASIBILITY_GO.json').read_text())
for r in fg['pins']:assert pin(r['path'])==r
assert json.loads((P/'BASELINE_REPLAY.json').read_text())['status']=='PASS'
schedule=json.loads((P/'SCHEDULE.json').read_text());prepared=json.loads((P/'PREPARED_INPUTS.json').read_text())
paths=[p for p in P.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json']]+[W/'output-support.npz']
for row in prepared['inputs']:
 for r in row['prepared_files']:assert pin(r['path'])==r;paths.append(Path(r['path']))
paths+=list((W/'baseline-crops').glob('*.npy'))
lookup={r['path']:r for r in old['pins']}
for p in paths:lookup[str(p)]=pin(p)
go=dict(status='FROZEN_FOR_SHIFTED_QUALIFICATION',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=old['models'],pins=list(lookup.values()),runtime_pins=old['runtime_pins'],core_xywh=schedule['scoring_xywh'],grids=schedule['grids'],depth_positions=prepared['depth_positions'],depths=[-15,15],batch_size=4,threads=4,precision='fp32 CPU',expected_maps=32,expected_batches=3096,qualification_max_forward_calls=288,qualification_max_tile_evaluations=768,gpu_spend_GBP=0)
save(P/'INPUT_GO.json',go)
print(json.dumps(dict(status=go['status'],pins=len(go['pins'])+len(go['runtime_pins'])+len(go['models']))))
