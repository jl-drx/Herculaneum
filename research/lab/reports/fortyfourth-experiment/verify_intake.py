"""Read-only independent source-body and raw-coordinate reconstruction."""
import io
import numpy as np
from PIL import Image
from pixel_http import *
for go in ['HEADER_GO.json','PIXEL_GO.json','PIXEL_EXECUTION_GO.json','TRAINING_GO.json']:
 for item in json.loads((P/go).read_text())['pins']:need(pin(item['path'])==item,'Frozen '+go)
ledgers={name:json.loads((P/name).read_text()) for name in ['HEADER_REQUESTS.json','PIXEL_REQUESTS.json']};sizes={}
for name,rows in ledgers.items():
 for r in rows:
  need(r['complete'] and 'error' not in r and pin(r['body']['path'])==r['body'],'Retained complete response identity')
  if r['status']==206:
   a,b=r['requested_range'];need(r['bytes']==b-a+1 and r['headers']['Content-Range']==f"bytes {a}-{b}/{r['object_bytes']}",'Strict byte range')
  for key in ['source_url','request_url','redirect_url']:
   if key in r:need(not urllib.parse.urlsplit(r[key]).query,'No signed query persisted')
 sizes[name]=dict(responses=len(rows),bytes=sum(r['bytes'] for r in rows),failed=0)
need(sizes['HEADER_REQUESTS.json']['responses']<=64 and sizes['HEADER_REQUESTS.json']['bytes']<=16*1024**2,'Header cap');need(sizes['PIXEL_REQUESTS.json']['responses']<=128 and sizes['PIXEL_REQUESTS.json']['bytes']<=384*1024**2,'Pixel cap')
selection=json.loads((P/'CROP_SELECTION.json').read_text());cases=json.loads((P/'HEADER_SELECTION.json').read_text())['cases'];checks=[]
for c,selected in zip(cases,selection):
 cid=c['case_id'];need(cid==selected['case'],'Stable case order');rows=ledgers['PIXEL_REQUESTS.json'];raw=next(r for r in rows if r['tag']==cid+'-raw-rows' and r['status']==206);x,y=selected['x'],selected['y'];s=selected['size']
 # Separate NumPy text parser, not the acquisition parser.
 body=Path(raw['body']['path']).read_bytes();reconstructed=np.loadtxt(io.BytesIO(body),delimiter=';',usecols=tuple(range(x+1,x+s+1)),dtype=np.float32);stored=np.load(P/'crops'/f'{cid}-height.npy');need(np.array_equal(reconstructed,stored,equal_nan=True),'Independent raw height reconstruction')
 coords=np.loadtxt(io.BytesIO(body),delimiter=';',usecols=(0,),dtype=np.float64);m=next(v for v in json.loads((P/'LAYOUTS_COMPLETE.json').read_text()) if v['case']==cid and v['kind']=='raw_height_txt');step=m['last_y']/(c['height']-1);need(np.allclose(coords,np.arange(y,y+s)*step,atol=.0002,rtol=0),'All raw coordinates contiguous')
 row=next(r for r in rows if r['tag']==cid+'-label-full' and r['status']==206);need(row['body']['sha256']==c['files']['label_png']['lfs']['oid'],'Full label object SHA256');labels=np.array(Image.open(row['body']['path']));need(np.array_equal(labels[y:y+s,x:x+s],np.load(P/'crops'/f'{cid}-label.npy')),'Label same-grid extraction')
 # Independent selection reconstruction with full labels and prescribed grid/ranking.
 grid=sorted((( (yy+s/2-c['height']/2)**2+(xx+s/2-c['width']/2)**2,yy,xx) for yy in range(256,c['height']-s-63,256) for xx in range(256,c['width']-s-63,256)))
 chosen=next((yy,xx) for _,yy,xx in grid if .05<=(labels[yy:yy+s,xx:xx+s]!=0).mean()<=.7);need(chosen==(y,x),'Frozen selection')
 # The original LFS resolver header binds every crop range to its published object identity.
 identity=[]
 for kind in c['files']:
  candidate=[r for r in rows if r['source_url'].endswith(urllib.parse.quote(c[kind],safe='/')) and r['status']==302]
  for r in candidate:
   etag=r['headers'].get('X-Linked-Etag');need(etag and etag.strip('"')==c['files'][kind]['lfs']['oid'],'Resolver linked LFS identity')
  identity.append(dict(kind=kind,resolver_responses=len(candidate),full_file_digest_checked=kind=='label_png'))
 checks.append(dict(case=cid,raw_values_reconstructed=int(stored.size),missing_values=int((~np.isfinite(stored)).sum()),coordinate_rows=s,label_LFS_verified=True,source_identity=identity))
save(P/'INTAKE_VERIFICATION.json',dict(status='PASS',created_at_utc=now(),ledger_totals=sizes,cases=checks,range_hash_scope='Retained range bodies verified; only label whole-object digests verified',full_raw_height_files_downloaded=False,source_TIFFs_modified=False,network_retries=0));print(json.dumps(dict(status='PASS',totals=sizes,cases=len(checks))))
