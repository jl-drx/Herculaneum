from pathlib import Path
import hashlib,json,importlib.util,sys
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
REPORTS=HERE.parent
PRIOR=REPORTS/'fortyfirst-experiment/research'
def need(ok,message):
 if not ok:raise ValueError(message)
def pin(path):
 path=Path(path).resolve(strict=True)
 with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(path),sha256=digest,size_bytes=path.stat().st_size)
def verify(p):need(pin(p['path'])==p,'Pin changed: '+p['path']);return Path(p['path'])
def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
