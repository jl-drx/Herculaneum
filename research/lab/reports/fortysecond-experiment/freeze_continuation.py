import ast
from common import *
prior=json.loads((HERE/'ALIGNMENT_GO.json').read_text());pins=prior['pins'][:]
for p in [HERE/'ALIGNMENT_INTERRUPTION.md',HERE/'complete_alignment.py',HERE/'freeze_continuation.py',*[p for p in (HERE/'alignment').glob('*.npy')]]:
 if p.suffix=='.py':ast.parse(p.read_text())
 pins.append(pin(p))
save(HERE/'ALIGNMENT_CONTINUATION_GO.json',dict(status='GO_COMPLETE_REMAINING_DIAGNOSTICS_UNCHANGED',created_at_utc=now(),pins=pins,no_new_acquisition=True,no_repeated_correlation_search=True,coverage_gate_unchanged=True))
print(json.dumps(pin(HERE/'ALIGNMENT_CONTINUATION_GO.json')))
