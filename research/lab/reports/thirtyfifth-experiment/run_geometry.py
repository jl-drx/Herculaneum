"""One exact PPM multipart request followed by all fixed regional diagnostics."""
import numpy as np
from geometry_common import *
from geometry_transport import Client
from geometry_math import describe
plan=json.loads((HERE/'GEOMETRY_PLAN.json').read_text());go=json.loads((HERE/'GEOMETRY_GO.json').read_text())
for item in go['files']:verify(item)
out=HERE/'geometry-v1';out.mkdir(exist_ok=False)
client=Client(out,plan)
try:
    path,entry=client.get(plan['object']['url'],plan['ranges'],'ppm-multipart.bin',plan['object'])
    payload=path.read_bytes();parts={tuple(p['range_inclusive']):p for p in entry['parts']};assembled=[]
    for span in plan['ranges']:
        p=parts[tuple(span)];block=payload[p['payload_offset']:p['payload_offset']+p['payload_bytes']]
        require(len(block)==144 and hashlib.sha256(block).hexdigest()==p['payload_sha256'],'PPM row identity');assembled.append(block)
    raw=b''.join(assembled);require(len(raw)==15552,'Fixed useful record extent');(out/'records.bin').write_bytes(raw)
    values=np.frombuffer(raw,dtype='<f8').reshape(36,3,3,6);records=[]
    for site,q in zip(plan['sites'],values,strict=True):records.append(dict(**site,geometry=describe(q)))
    for item in go['files']:verify(item)
    save(out/'COMPLETE.json',dict(status='PASS',created_at_utc=now(),GO=pin(HERE/'GEOMETRY_GO.json'),ledger=pin(out/'LEDGER.json'),
        raw_records=pin(out/'records.bin'),sites=records,requests=len(client.ledger['requests']),returned_body_bytes=client.ledger['returned_body_bytes'],
        reserved_body_bytes=client.ledger['reserved_body_bytes'],active_acquisition_seconds=client.ledger['active_acquisition_seconds'],
        peak_RSS_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024,
        new_CT_requests=0,new_model_forwards=0,GPU_spend_GBP=0,
        scope='Conditional PPM serialization/local geometry and declared ray bounds; not optical/anatomical qualification'))
    print(json.dumps(dict(status='PASS',sites=len(records),mapped_records=sum(r['geometry']['mapped_records'] for r in records),complete_sites=sum(r['geometry']['complete'] for r in records),
        requests=len(client.ledger['requests']),returned_body_bytes=client.ledger['returned_body_bytes'])))
except Exception as e:
    save(out/'FAILURE.json',dict(status='STOP',error=type(e).__name__+': '+str(e)));raise
