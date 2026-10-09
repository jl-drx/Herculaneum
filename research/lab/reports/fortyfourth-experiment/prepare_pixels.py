from intake_http import *
import numpy as np
cases=json.loads((P/'headers/case-manifest-hop0.bin').read_text())['samples'];tree=json.loads((P/'headers/data-tree-hop0.bin').read_text());splits=json.loads((P/'headers/splits-hop0.bin').read_text());ids={c['case_id'] for c in cases};parents={c['case_id']:c['papyrus_id'] for c in cases}
expected={c[k] for c in cases for k in ['raw_height_txt','normalized_height_tif','brightfield_tif','label_png']};actual={f['path'] for f in tree if f['type']=='file'};need(expected==actual and len(expected)==56,'Exact file bijection')
sr=[]
for i,s in enumerate(splits):
 a,b=set(s['train']),set(s['val']);need(not a&b and a|b==ids and len(a)==len(s['train']) and len(b)==len(s['val']),'Fold disjoint exhaustive')
 ap={parents[v] for v in a};bp={parents[v] for v in b};disjoint=not ap&bp
 if i>=5:need(disjoint and len(bp)==1,'Leave-parent-out fold')
 sr.append(dict(fold=i,train=len(a),validation=len(b),parent_disjoint=disjoint,held_out_parents=sorted(bp)))
layouts=json.loads((P/'LAYOUTS_COMPLETE.json').read_text());cs=json.loads((P/'HEADER_SELECTION.json').read_text())['cases']
for c in cs:
 for k in ['normalized_height_tif','brightfield_tif']:
  r=next(x for x in layouts if x['case']==c['case_id'] and x['kind']==k);need(r['shape'][:2]==[c['height'],c['width']] and r['orientation']==1,'TIFF common shape and origin')
 r=next(x for x in layouts if x['case']==c['case_id'] and x['kind']=='raw_height_txt');need(r['x_header_values']==r['last_row_values']==c['width'] and abs(r['last_y']/0.68793625-(c['height']-1))<0.01,'Raw coordinate counts')
ledger=json.loads((P/'HEADER_REQUESTS.json').read_text());need(all(r['complete'] and 'error' not in r for r in ledger),'Header requests complete')
save(P/'HEADER_VERIFICATION.json',dict(status='PASS',responses=len(ledger),bytes=sum(r['bytes'] for r in ledger),expected_files=56,folds=sr,header_failures=['Six TIFF head-only seeks exceeded available range; resolved with retained tail ranges','First raw row parser rejected final empty field; one documented trailing delimiter removed in corrected parser'],brightfield_issue='All three RGB TIFFs specify eight-bit IEEEFP; display interpretation must be explicit'))
source=(P/'intake_http.py').read_text().replace("P/'HEADER_GO.json'","P/'PIXEL_GO.json'").replace("P/'headers'","P/'pixel-responses'").replace("P/'HEADER_REQUESTS.json'","P/'PIXEL_REQUESTS.json'").replace('len(rows)<64','len(rows)<128').replace('16*1024**2','384*1024**2').replace('128*1024','96*1024**2').replace('Header','Pixel').replace('Metadata must','Non-ranged request must')
(P/'pixel_http.py').write_text(source)
save(P/'PIXEL_GO.json',dict(created_at_utc=now(),status='GO_FOR_FIXED_PIXEL_INTAKE_ONLY',pins=[pin(P/n) for n in ['PIXEL_PROTOCOL.md','HEADER_SELECTION.json','LAYOUTS_COMPLETE.json','HEADER_VERIFICATION.json','pixel_http.py','prepare_pixels.py']],training_authorized=False))
print(json.dumps(dict(status='PASS',header_responses=len(ledger),header_bytes=sum(r['bytes'] for r in ledger),folds=sr)))
