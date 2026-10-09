"""Geometry-only admission with strict delivered-byte accounting and ranged TIFF I/O."""
import io,math,time,urllib.request,urllib.error
import numpy as np
import tifffile
from common import *
BASE='https://vesuvius-challenge-open-data.s3.amazonaws.com/'
OUT=HERE/'geometry'
CAP=32*1024**2
class Client:
 def __init__(self):self.records=[];self.delivered=0;self.started=time.monotonic()
 def get(self,url,tag,size,start=None,length=None):
  expected=size if start is None else length
  need(self.delivered+expected<=CAP,'Geometry body cap')
  headers={'Accept-Encoding':'identity'}
  if start is not None:headers['Range']=f'bytes={start}-{start+length-1}'
  req=urllib.request.Request(url,headers=headers);record=dict(url=url,tag=tag,start=start,requested_bytes=expected)
  try:
   with urllib.request.urlopen(req,timeout=30) as r:
    record.update(status=r.status,content_range=r.headers.get('Content-Range'),content_length=r.headers.get('Content-Length'),etag=r.headers.get('ETag'))
    if start is None:need(r.status==200,'Full GET status')
    else:need(r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{start+length-1}/{size}','Exact HTTP range response')
    body=r.read(expected+1)
    need(len(body)==expected,'Exact response length')
   self.delivered+=len(body);path=OUT/f'{len(self.records):03d}-{tag}.bin';path.write_bytes(body);record.update(body=pin(path),bytes=len(body),elapsed_seconds=time.monotonic()-self.started)
   self.records.append(record);self.flush();return body
  except Exception as e:
   record.update(error=type(e).__name__+': '+str(e));self.records.append(record);self.flush();raise
 def flush(self):
  (OUT/'REQUESTS.json').write_text(json.dumps(dict(records=self.records,delivered_bytes=self.delivered,cap_bytes=CAP),indent=2)+'\n')
class Remote(io.RawIOBase):
 def __init__(self,client,url,size,tag):self.client=client;self.url=url;self.size=size;self.tag=tag;self.position=0;self.cache={};self.name=tag
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.position
 def seek(self,offset,whence=0):
  p=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
  need(0<=p<=self.size,'Bounded TIFF seek');self.position=p;return p
 def read(self,n=-1):
  need(n>=0,'No unbounded TIFF read');n=min(n,self.size-self.position);out=[]
  while n:
   k=self.position//65536;begin=k*65536
   if k not in self.cache:self.cache[k]=self.client.get(self.url,self.tag+f'-block{k}',self.size,begin,min(65536,self.size-begin))
   off=self.position-begin;part=self.cache[k][off:off+n];out.append(part);self.position+=len(part);n-=len(part)
  return b''.join(out)
def values(remote,page,rows,cols):
 need(page.compression==1 and page.dtype==np.dtype('float32') and len(page.shape)==2 and not page.is_tiled,'Uncompressed float32 strip layout')
 h,w=page.shape;out=[]
 for y,x in zip(rows,cols):
  need(0<=y<h and 0<=x<w,'Sample within TIFF')
  strip=int(y)//page.rowsperstrip;offset=int(page.dataoffsets[strip])+((int(y)%page.rowsperstrip)*w+int(x))*4
  remote.seek(offset);out.append(np.frombuffer(remote.read(4),dtype=page.dtype.newbyteorder(page.parent.byteorder))[0])
 return np.asarray(out,np.float64)
def main():
 go=json.loads((HERE/'GEOMETRY_GO.json').read_text())
 for item in go['pins']:verify(item)
 OUT.mkdir(exist_ok=False);client=Client();meta=json.loads((PRIOR/'PAIR_FEASIBILITY.json').read_text());old=[];descriptions=[]
 root='PHercParis4/segments/20231016151002/mesh/'
 old_sizes=[1009996,987620,810856]
 for axis,size in zip('xyz',old_sizes):
  body=client.get(BASE+root+f'20231016151002-on-20230205180739-7.91um.tifxyz/{axis}.tif','old-'+axis,size)
  a=tifffile.imread(io.BytesIO(body));need(a.ndim==2 and a.dtype==np.float32,'Old float32 grid');old.append(a)
 old=np.stack(old,-1);h,w=old.shape[:2];valid=np.isfinite(old).all(-1)&(old!=0).any(-1)&(old>0).all(-1)&(old<np.asarray(meta['old_shape_ZYX'][::-1])).all(-1)
 from scipy.ndimage import minimum_filter
 valid=minimum_filter(valid.astype(np.uint8),size=5,mode='constant',cval=0)>0
 yy,xx=np.where(valid);candidates=[];selected=[]
 for i,(fx,fy) in enumerate([(x,y) for y in [.2,.5,.8] for x in [.3,.7]]):
  target=np.array([fy*(h-1),fx*(w-1)]);eligible=(abs(yy-target[0])<=.08*h)&(abs(xx-target[1])<=.08*w)
  ey,ex=yy[eligible],xx[eligible];need(len(ey)>0,'No geometric candidate in frozen neighborhood')
  d=((ey-target[0])/h)**2+((ex-target[1])/w)**2;j=np.lexsort((ex,ey,d))[0];y,x=int(ey[j]),int(ex[j]);point=old[y,x].astype(float)
  need(all(np.linalg.norm(point-other)*7.91>=1000 for other in selected),'Physical center separation >=1mm');selected.append(point)
  modern=np.asarray(meta['forward']['matrix'])@np.r_[point,1]
  candidates.append(dict(id=f'pair{i+1:02d}',nominal_role=['training','training','selection','selection','comparison','comparison'][i],normalized_target_xy=[fx,fy],old_grid_yx=[y,x],old_xyz=point.tolist(),new_xyz_published_transform=modern.tolist()))
 # Compare normalized-grid locations as a declared diagnostic, not assumed correspondence.
 newvalues=[];layout=[]
 for axis in 'xyz':
  remote=Remote(client,BASE+root+f'20231016151002-on-20260411134726-2.4um.tifxyz/{axis}.tif',16606550,'new-'+axis)
  with tifffile.TiffFile(remote) as tf:
   page=tf.pages[0];nh,nw=page.shape;ys=[int(round(c['old_grid_yx'][0]*(nh-1)/(h-1))) for c in candidates];xs=[int(round(c['old_grid_yx'][1]*(nw-1)/(w-1))) for c in candidates]
   newvalues.append(values(remote,page,ys,xs));layout.append(dict(axis=axis,shape=list(page.shape),dtype=str(page.dtype),compression=int(page.compression),rowsperstrip=page.rowsperstrip,strip_count=len(page.dataoffsets),sample_yx=list(map(list,zip(ys,xs)))))
 for c,v in zip(candidates,np.stack(newvalues,-1)):
  c['new_grid_normalized_sample_xyz']=v.tolist();c['normalized_grid_discrepancy_new_voxels']=float(np.linalg.norm(v-c['new_xyz_published_transform']))
 np.save(OUT/'old-xyz.npy',old,allow_pickle=False)
 save(OUT/'SELECTION.json',dict(status='GEOMETRY_CENTERS_SELECTED',old_shape=list(old.shape),new_layout=layout,candidates=candidates,selection_before_CT_and_labels=True,all_valid_old_samples=int(valid.sum()),geometry_bytes=client.delivered,physical_alignment_established=False,normalized_grid_relation='diagnostic only; not used to choose centers'))
 save(OUT/'COMPLETE.json',dict(status='PASS_GEOMETRY_ACQUISITION',created_at_utc=now(),requests=len(client.records),delivered_bytes=client.delivered,old_array=pin(OUT/'old-xyz.npy'),selection=pin(OUT/'SELECTION.json'),records=pin(OUT/'REQUESTS.json'),new_CT_reads=0,new_label_reads=0,new_model_calls=0))
 print((OUT/'SELECTION.json').read_text())
if __name__=='__main__':main()
