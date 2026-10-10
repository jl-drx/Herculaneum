"""Verify the compact handoff before independent implementation; stdlib only."""
import hashlib,json
from pathlib import Path,PurePosixPath
P=Path(__file__).resolve().parent;m=json.loads((P/'PACKET_MANIFEST.json').read_text());seen=set()
for r in m['files']:
 name=r['path'];rel=PurePosixPath(name)
 if rel.is_absolute() or '..' in rel.parts or name in seen:raise ValueError('Unsafe/duplicate path')
 seen.add(name);p=P/name
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=r['bytes']:raise ValueError('Missing, linked or wrong-sized member: '+name)
 with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 if sha!=r['sha256']:raise ValueError('Checksum mismatch: '+name)
print(json.dumps({'status':'PASS','verified_files':len(seen),'inference_run':False}))
