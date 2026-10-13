from pathlib import Path
import hashlib,json,os
P=Path(__file__).resolve().parent;W=P/'work';R=P.parent;LAB=P.parents[1]
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h)
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def atomjson(p,v):
 p=Path(p);t=p.with_name(p.name+'.partial')
 with t.open('w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def atomnpy(p,a):
 import numpy as np
 p=Path(p);t=p.with_name(p.name+'.partial')
 with t.open('wb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
