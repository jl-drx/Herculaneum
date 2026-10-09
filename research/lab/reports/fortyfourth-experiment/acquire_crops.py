"""Fixed paired crops from pinned public source; no model or image-driven selection."""
import io,math
import numpy as np
from PIL import Image
from pixel_http import *
S=1536
cases=json.loads((P/'HEADER_SELECTION.json').read_text())['cases']
layouts=json.loads((P/'LAYOUTS_COMPLETE.json').read_text())
def fields(line):
 v=line.rstrip(b'\r\n').split(b';')
 return v[:-1] if v[-1]==b'' else v
def get(c,kind,tag,a,b):
 path,row=fetch(c['case_id']+'-'+tag,f'https://huggingface.co/datasets/scrollprize/profilometer/resolve/{REV}/'+urllib.parse.quote(c[kind],safe='/'),(a,b))
 need(row['object_bytes']==c['files'][kind]['size'],'Source size identity')
 return path.read_bytes()
def image_crop(c,kind,y,x):
 m=next(v for v in layouts if v['case']==c['case_id'] and v['kind']==kind);need(m['compression']==1 and m['planarconfig']==1,'Uncompressed interleaved strips')
 channels=3 if kind=='brightfield_tif' else 1;dtype=np.dtype('u1' if channels==3 else '<u2');need(m['bits_per_sample']==([8,8,8] if channels==3 else 16),'Explicit stored bits')
 r=m['rowsperstrip'];first=y//r;last=(y+S-1)//r;offs=m['offsets'];counts=m['bytecounts']
 for i in range(first,last):need(offs[i]+counts[i]==offs[i+1],'Contiguous strip range')
 for i in range(first,last+1):need(counts[i]==min(r,c['height']-i*r)*c['width']*channels*dtype.itemsize,'Strip byte geometry')
 a=offs[first];b=offs[last]+counts[last]-1;body=get(c,kind,kind+'-crop',a,b);arr=np.frombuffer(body,dtype).reshape(-1,c['width'],channels);v=arr[y-first*r:y-first*r+S,x:x+S].copy()
 return v[:,:,0] if channels==1 else v

def raw_crop(c,y,x):
 m=next(v for v in layouts if v['case']==c['case_id'] and v['kind']=='raw_height_txt');first=Path(m['head']['path']).read_bytes();header=first.splitlines(keepends=True)[0];xs=np.array([float(v) for v in fields(header)[1:]]);step=m['last_y']/(c['height']-1);need(abs(step-np.median(np.diff(xs)))<0.0002,'XY coordinate increments')
 H=c['height'];size=m['size'];probes=[]
 def rows_at(start,body):
  lines=body.splitlines(keepends=True);offset=start;out=[]
  for j,line in enumerate(lines):
   pos=offset;offset+=len(line)
   if start!=0 and j==0:continue
   if not line.endswith(b'\n') and offset<size:continue
   f=fields(line)
   if not f[0]:continue
   coord=float(f[0]);index=round(coord/step)
   need(abs(coord-index*step)<0.0002 and 0<=index<H and len(f)==c['width']+1,'Raw row coordinate and width')
   out.append((index,pos,offset))
  need(bool(out),'At least one complete raw row in probe');return out
 def boundary(target):
  if target==H:return size
  if target==0:return len(header)
  low_i,low_b=0,len(header);high_i,high_b=H,size
  for count in range(16):
   estimate=round(low_b+(target-low_i)/(high_i-low_i)*(high_b-low_b));a=max(low_b,estimate-65536);a=min(a,max(0,size-524288));b=min(size-1,a+524288-1)
   body=get(c,'raw_height_txt',f'raw-probe-{target}-{count}',a,b);rr=rows_at(a,body);probes.append(dict(target=target,range=[a,b],first_row=rr[0][0],last_row=rr[-1][0]))
   for index,pos,end in rr:
    if index==target:return pos
   if rr[-1][0]<target:low_i,low_b=rr[-1][:2]
   elif rr[0][0]>target:high_i,high_b=rr[0][:2]
   else:raise ValueError('Noncontiguous raw probe')
  raise ValueError('Raw boundary search cap')
 a=boundary(y);b=boundary(y+S);need(0<b-a<=96*1024**2,'Raw row block bound');body=get(c,'raw_height_txt','raw-rows',a,b-1);lines=body.splitlines();need(len(lines)==S,'Exact raw row block')
 z=np.empty((S,S),np.float32)
 for i,line in enumerate(lines):
  f=fields(line);need(len(f)==c['width']+1 and abs(float(f[0])-(y+i)*step)<0.0002,'Raw crop row position')
  values=np.array([float(v) if v else np.nan for v in f[1:]],np.float64);need(not np.isinf(values).any(),'No infinities');z[i]=values[x:x+S]
 return z,dict(probes=probes,raw_byte_range=[a,b-1],coordinate_step=step)

def main():
 for p in json.loads((P/'PIXEL_EXECUTION_GO.json').read_text())['pins']:need(pin(p['path'])==p,'Execution pin')
 (P/'crops').mkdir(exist_ok=True);selection=[]
 for c in cases:
  size=c['files']['label_png']['size'];body=get(c,'label_png','label-full',0,size-1);need(hashlib.sha256(body).hexdigest()==c['files']['label_png']['lfs']['oid'],'Full label LFS hash')
  label=np.array(Image.open(io.BytesIO(body)));need(label.shape==(c['height'],c['width']) and label.ndim==2,'Label geometry');binary=label!=0
  grid=sorted((( (y+S/2-c['height']/2)**2+(x+S/2-c['width']/2)**2,y,x) for y in range(256,c['height']-S-63,256) for x in range(256,c['width']-S-63,256)))
  qualifying=[(y,x,float(binary[y:y+S,x:x+S].mean())) for _,y,x in grid if .05<=binary[y:y+S,x:x+S].mean()<=.7];need(bool(qualifying),'Crop class-support selection');y,x,frac=qualifying[0]
  item=dict(case=c['case_id'],parent=c['papyrus_id'],x=x,y=y,size=S,source_label_values=np.unique(label).tolist(),annotation_foreground_fraction=frac);selection.append(item);save(P/'crops'/f"{c['case_id']}-selection.json",item)
  print(json.dumps(dict(selected=item)),flush=True)
  np.save(P/'crops'/f"{c['case_id']}-label.npy",label[y:y+S,x:x+S]);np.save(P/'crops'/f"{c['case_id']}-rgb.npy",image_crop(c,'brightfield_tif',y,x));np.save(P/'crops'/f"{c['case_id']}-normalized.npy",image_crop(c,'normalized_height_tif',y,x));z,meta=raw_crop(c,y,x);np.save(P/'crops'/f"{c['case_id']}-height.npy",z);save(P/'crops'/f"{c['case_id']}-raw-layout.json",meta)
 save(P/'CROP_SELECTION.json',selection)
if __name__=='__main__':main()
