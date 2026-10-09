import ast
from common import *
for p in HERE.glob('*.py'):ast.parse(p.read_text())
paths=[HERE/'PROTOCOL.md',HERE/'common.py',HERE/'acquire_geometry.py',HERE/'freeze_geometry.py',PRIOR/'PAIR_FEASIBILITY.json',PRIOR/'pair-registration-metadata/PHercParis4-catalog.json',PRIOR/'pair-final-metadata/old-native-array.txt',PRIOR/'pair-final-metadata/new-native-array.txt',PRIOR/'pair-final-metadata/catalog-schema.txt']
save(HERE/'GEOMETRY_GO.json',dict(status='GO_CPU_GEOMETRY_ONLY',created_at_utc=now(),user_authorization='Proceed with phase41 recommendation',pins=[pin(p) for p in paths],geometry_reference_cap_bytes=32*1024**2,CT_chunk_cap_bytes=256*1024**2,model_calls_permitted=0,scientific_payload_calls_before_GO=0))
print(json.dumps(pin(HERE/'GEOMETRY_GO.json')))
