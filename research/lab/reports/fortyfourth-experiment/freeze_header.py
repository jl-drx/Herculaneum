from intake_http import *
inputs=[P/'PROTOCOL.md',P/'intake_http.py',P/'freeze_header.py',P.parent/'fortythird-experiment/source-evidence/profilometer-manifest.csv',P.parent/'fortythird-experiment/source-evidence/profilometer-experiment.json',P.parent/'fortythird-experiment/final-verification.json']
save(P/'HEADER_GO.json',dict(status='GO_PINNED_METADATA_AND_HEADERS_ONLY',created_at_utc=now(),revision=REV,pins=[pin(p) for p in inputs],maximum_responses=64,maximum_bytes=16*1024**2,maximum_header_range=128*1024,model_calls_permitted=0))
