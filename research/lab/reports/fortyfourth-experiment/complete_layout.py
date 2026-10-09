"""Read tail-resident TIFF directories and explicit coordinate-matrix boundaries."""
import io,struct
import numpy as np
import tifffile
from intake_http import *
class Sparse(io.RawIOBase):
 def __init__(self,size,ranges):self.size=size;self.ranges=ranges;self.pos=0;self.name='bounded-header'
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,n,whence=0):
  self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
  need(self.pos>=0,'Nonnegative seek');return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n==0:return b''
  for start,body in self.ranges:
   if start<=self.pos and self.pos+n<=start+len(body):out=body[self.pos-start:self.pos-start+n];self.pos+=n;return out
  raise ValueError('Attempt to read unacquired file range')
def main():
 selection=json.loads((P/'HEADER_SELECTION.json').read_text());rows=[]
 for c in selection['cases']:
  for kind in ['normalized_height_tif','brightfield_tif','raw_height_txt']:
   previous=json.loads((P/'HEADER_REQUESTS.json').read_text());head=next(r for r in previous if r['tag']==c['case_id']+'-'+kind and r['status']==206);first=Path(head['body']['path']).read_bytes();size=c['files'][kind]['size'];start=max(0,size-65536)
   path,r=fetch(c['case_id']+'-'+kind+'-tail',f'https://huggingface.co/datasets/scrollprize/profilometer/resolve/{REV}/'+urllib.parse.quote(c[kind],safe='/'),(start,size-1));tail=path.read_bytes();need(r['object_bytes']==size,'Pinned size')
   meta=dict(case=c['case_id'],kind=kind,size=size,head=head['body'],tail=pin(path),tail_start=start)
   if kind.endswith('tif'):
    with tifffile.TiffFile(Sparse(size,[(0,first),(start,tail)])) as t:
     page=t.pages[0];meta.update(shape=list(page.shape),dtype=str(page.dtype),byteorder=t.byteorder,compression=int(page.compression),photometric=int(page.photometric),planarconfig=int(page.planarconfig),rowsperstrip=page.rowsperstrip,offsets=list(page.dataoffsets),bytecounts=list(page.databytecounts),orientation=int(page.tags['Orientation'].value) if 'Orientation' in page.tags else None)
   else:
    lines=first.splitlines();x=np.array([float(s) for s in lines[0].split(b';')[1:]]);last=tail.splitlines()[-1];last_values=last.split(b';');y=float(last_values[0]);meta.update(x_header_values=len(x),x_first=float(x[0]),x_last=float(x[-1]),first_y=float(lines[1].split(b';',1)[0]),last_y=y,last_row_values=len(last_values)-1,header_bytes=len(lines[0])+2)
   rows.append(meta);print(json.dumps({k:v for k,v in meta.items() if k not in ['head','tail','offsets','bytecounts']},ensure_ascii=False),flush=True)
 save(P/'LAYOUTS_COMPLETE.json',rows)
if __name__=='__main__':main()
