import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
assert not (BASE/'controller-start.json').exists(), 'No retry'
freeze=json.loads((BASE/'SOURCE_FREEZE.json').read_text())
for r in freeze['sources']:
    b=Path(r['path']).read_bytes()
    assert len(b)==r['byte_length'] and hashlib.sha256(b).hexdigest()==r['sha256']
(BASE/'controller-start.json').write_text(json.dumps({'source_freeze_sha256':hashlib.sha256((BASE/'SOURCE_FREEZE.json').read_bytes()).hexdigest(),'wall_limit_seconds':210,'monotonic_start':time.monotonic()},indent=2)+'\n')
start=time.monotonic()
try:
    p=subprocess.run([sys.executable,str(BASE/'bounded_metadata.py'),str(BASE/'run-v1')],timeout=210,capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    out,err,code=p.stdout,p.stderr,p.returncode
    status='PASS' if code==0 else 'STOP'
except subprocess.TimeoutExpired as e:
    out,err,code=e.stdout or b'',e.stderr or b'',None
    status='STOP_EXTERNAL_TIMEOUT'
(BASE/'worker.stdout').write_bytes(out)
(BASE/'worker.stderr').write_bytes(err)
body_files=list((BASE/'run-v1').glob('*.body'))
result={'status':status,'exit_code':code,'wall_seconds':time.monotonic()-start,'retained_body_bytes':sum(p.stat().st_size for p in body_files),'body_file_count':len(body_files)}
(BASE/'controller-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(out.decode(errors='replace'))
print(json.dumps(result))
sys.exit(0 if status=='PASS' else 1)
