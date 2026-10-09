"""Resume retained tails after diagnosing a trailing semicolon in raw rows."""
from complete_layout import *
def fields(line):
 v=line.rstrip(b'\r\n').split(b';')
 if v[-1]==b'':v=v[:-1]
 return v
selection=json.loads((P/'HEADER_SELECTION.json').read_text());rows=[]
for c in selection['cases']:
 for kind in ['normalized_height_tif','brightfield_tif','raw_height_txt']:
  previous=json.loads((P/'HEADER_REQUESTS.json').read_text());head=next(r for r in previous if r['tag']==c['case_id']+'-'+kind and r['status']==206);first=Path(head['body']['path']).read_bytes();size=c['files'][kind]['size'];start=max(0,size-65536);tag=c['case_id']+'-'+kind+'-tail'
  matches=[r for r in previous if r['tag']==tag and r['status']==206 and r['complete']]
  if matches:r=matches[0];path=Path(r['body']['path']);need(pin(path)==r['body'],'Cached tail hash')
  else:path,r=fetch(tag,f'https://huggingface.co/datasets/scrollprize/profilometer/resolve/{REV}/'+urllib.parse.quote(c[kind],safe='/'),(start,size-1))
  tail=path.read_bytes();need(r['object_bytes']==size,'Pinned size')
  meta=dict(case=c['case_id'],kind=kind,size=size,head=head['body'],tail=pin(path),tail_start=start)
  if kind.endswith('tif'):
   with tifffile.TiffFile(Sparse(size,[(0,first),(start,tail)])) as t:
    page=t.pages[0];meta.update(shape=list(page.shape),dtype=str(page.dtype),byteorder=t.byteorder,compression=int(page.compression),photometric=int(page.photometric),planarconfig=int(page.planarconfig),rowsperstrip=page.rowsperstrip,offsets=list(page.dataoffsets),bytecounts=list(page.databytecounts),orientation=int(page.tags['Orientation'].value) if 'Orientation' in page.tags else None,bits_per_sample=page.tags['BitsPerSample'].value,sample_format=page.tags['SampleFormat'].value if 'SampleFormat' in page.tags else None)
  else:
   lines=first.splitlines();x=np.array([float(s) for s in fields(lines[0])[1:]]);last_values=fields(tail.splitlines()[-1]);meta.update(x_header_values=len(x),x_first=float(x[0]),x_last=float(x[-1]),x_strict_increasing=bool(np.all(np.diff(x)>0)),first_y=float(fields(lines[1])[0]),last_y=float(last_values[0]),last_row_values=len(last_values)-1,header_bytes=len(lines[0])+2,trailing_empty_semicolon=True)
  rows.append(meta);print(json.dumps({k:v for k,v in meta.items() if k not in ['head','tail','offsets','bytecounts']},ensure_ascii=False),flush=True)
save(P/'LAYOUTS_COMPLETE.json',rows)
