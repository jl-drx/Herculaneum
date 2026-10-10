"""Phase32 bounded, immutable-artifact helpers and pinned legacy parsers."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
HERE=Path(__file__).resolve().parent
REPORTS=HERE.parent
LAB=REPORTS.parent
sys.path.insert(0,str(REPORTS/'sixteenth-experiment/production-acquisition'))
from production_common import parse_tiff, tiff_header, parse_multipart, content_range

def require(ok,why):
    if not ok:raise ValueError(why)

def pin(path):
    path=Path(path).resolve()
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)

def verify(item):
    require(pin(item['path'])==item,'Pinned bytes changed: '+item['path'])
    return Path(item['path'])

def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.writing')
    with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    tmp.replace(path)

def now():return datetime.now(timezone.utc).isoformat()

def check_freeze(name):
    frozen=json.loads((HERE/name).read_text())
    for item in frozen['files']:verify(item)
    return frozen
