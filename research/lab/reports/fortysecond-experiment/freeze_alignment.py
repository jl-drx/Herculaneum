import ast
from common import *
fixture=json.loads((HERE/'ALIGNMENT_TESTS.json').read_text());need(fixture['passed']==fixture['tests_run']==3,'Diagnostic fixtures')
paths=[HERE/x for x in ['PROTOCOL.md','ALIGNMENT_PROTOCOL.md','common.py','alignment.py','test_alignment.py','ALIGNMENT_TESTS.json','freeze_alignment.py','geometry/SELECTION.json','geometry/old-xyz.npy','ct/COMPLETE.json']]+[PRIOR/'PAIR_FEASIBILITY.json']
paths+=[Path(r['array']['path']) for r in json.loads((HERE/'ct/COMPLETE.json').read_text())['arrays']]
for p in paths:
 if p.suffix=='.py':ast.parse(p.read_text())
save(HERE/'ALIGNMENT_GO.json',dict(status='GO_STRUCTURAL_DIAGNOSTIC_ONLY',created_at_utc=now(),pins=[pin(p) for p in paths],new_model_calls_allowed=0,independent_ink_reference_established=False))
print(json.dumps(pin(HERE/'ALIGNMENT_GO.json')))
