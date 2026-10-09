from pixel_http import *
rows=json.loads((P/'PIXEL_REQUESTS.json').read_text());cs=json.loads((P/'HEADER_SELECTION.json').read_text())['cases'];crops=json.loads((P/'CROP_SELECTION.json').read_text());plans=[]
for c,s in zip(cs,crops):
 sources=[]
 for kind,tag in [('raw_height_txt','raw-rows'),('label_png','label-full')]:
  r=next(r for r in rows if r['tag']==c['case_id']+'-'+tag and r['status']==206);sources.append(dict(kind=kind,url=r['source_url'],byte_range=r['requested_range'],object_bytes=r['object_bytes'],LFS_sha256=c['files'][kind]['lfs']['oid'],range_sha256=r['body']['sha256'],range_bytes=r['bytes'],cache_file=Path(r['body']['path']).name))
 layout=json.loads((P/'crops'/f"{c['case_id']}-raw-layout.json").read_text());outputs={}
 for name in ['height','label','common-support']:
  r=pin(P/'crops'/f"{c['case_id']}-{name}.npy");outputs[name]={k:v for k,v in r.items() if k!='path'}
 plans.append(dict(selection=s,full_shape=[c['height'],c['width']],coordinate_step=layout['coordinate_step'],sources=sources,outputs=outputs))
save(P/'CROP_REPLAY_PLAN.json',dict(schema_version=1,source_revision=REV,license='CC-BY-NC-4.0',attribution='Vesuvius Challenge (2026). Optical profilometry of opened Herculaneum papyri. https://doi.org/10.1038/s41598-026-58467-1.',cases=plans,scope='Materialize original fixed training inputs from immutable public ranges; no automatic training, no CT data',expected_delivered_payload_bytes=sum(r['range_bytes'] for p in plans for r in p['sources'])))
