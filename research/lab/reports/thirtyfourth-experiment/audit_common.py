from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
HERE=Path(__file__).resolve().parent
REPORTS=HERE.parent

def require(ok,why):
    if not ok:raise ValueError(why)
def pin(path):
    path=Path(path).resolve()
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)
def verify(item):
    require(pin(item['path'])==item,'Pinned input changed: '+item['path']);return Path(item['path'])
def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.writing')
    with temp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    temp.replace(path)
def now():return datetime.now(timezone.utc).isoformat()
def check_freeze():
    go=json.loads((HERE/'GO.json').read_text())
    for p in go['files']:verify(p)
    return go
