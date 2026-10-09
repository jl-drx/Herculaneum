"""Metadata-only paired-acquisition feasibility, no image/label chunks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request,urllib.parse,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent/'pair-metadata-followup';ROOT.mkdir(exist_ok=False)
BASE='https://vesuvius-challenge-open-data.s3.amazonaws.com/'
records=[]
def get(tag,url):
 if len(records)>=24:raise RuntimeError('24 request cap')
 with urllib.request.urlopen(url,timeout=20) as r:
  body=r.read(1024*1024+1)
  if len(body)>1024*1024:raise RuntimeError('1MiB response cap')
  if sum(x['bytes'] for x in records)+len(body)>8*1024**2:raise RuntimeError('8MiB total cap')
 (ROOT/(tag+'.txt')).write_bytes(body)
 records.append(dict(tag=tag,url=url,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),retrieved_at_utc=datetime.now(timezone.utc).isoformat()))
 return body
def listing(tag,prefix):
 xml=get(tag,BASE+'?'+urllib.parse.urlencode({'list-type':2,'prefix':prefix,'delimiter':'/','max-keys':1000}))
 tree=ET.fromstring(xml);ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
 result=dict(prefix=prefix,prefixes=[x.text for x in tree.findall('s:CommonPrefixes/s:Prefix',ns)],objects=[x.text for x in tree.findall('s:Contents/s:Key',ns)],truncated=tree.findtext('s:IsTruncated',namespaces=ns))
 print(json.dumps(result),flush=True);return result
results=[]
try:
 for tag,prefix in [('root',''),('legacy-scroll1','full-scrolls/Scroll1/'),('mesh','PHercParis4/segments/20231016151002/mesh/'),('labels','PHercParis4/segments/20231016151002/ink-labels/')]:results.append(listing(tag,prefix))
 for tag,key in [('surface-attrs','PHercParis4/segments/20231016151002/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs'),('native-attrs','PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/.zattrs')]:
  try:
   data=json.loads(get(tag,BASE+key));print(json.dumps(dict(tag=tag,metadata=data)),flush=True);results.append(dict(tag=tag,metadata=data))
  except Exception as e:results.append(dict(tag=tag,error=str(e)))
except Exception as e:results.append(dict(error=str(e)))
(ROOT/'RECEIPT.json').write_text(json.dumps(dict(records=records,results=results,requests=len(records),bytes=sum(x['bytes'] for x in records),scientific_chunk_reads=0),indent=2)+'\n')
