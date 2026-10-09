"""Freeze reference-search scope before the first new request."""
from fetch_metadata import P,pin,now
import json
inputs=[P/'PROTOCOL.md',P/'fetch_metadata.py',P/'freeze.py',P.parent/'fortysecond-experiment/NEXT_STUDY_DRAFT.md',P.parent/'fortysecond-experiment/final-verification.json',P.parents[1]/'data/methods-review/paper.html',P.parents[1]/'data/methods-review/paper.txt',P.parents[1]/'data/methods-review/paper-code-tree.json']
go=dict(status='GO_METADATA_REFERENCE_SEARCH_ONLY',created_at_utc=now(),targets=['PHercParis4 direct-visible-ink reference','PHerc500P2 Extended Data Figure5 infrared reference'],max_requests=24,max_delivered_bytes=16*1024**2,max_response_bytes=4*1024**2,max_minutes=30,CT_reads_permitted=0,model_calls_permitted=0,pins=[pin(p) for p in inputs])
with (P/'GO.json').open('x') as f:json.dump(go,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(status=go['status'],pins=len(go['pins']))))
