"""Verify and restore the completed-results bundle into a new directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ap=argparse.ArgumentParser()
ap.add_argument('--bundle',type=Path,required=True)
ap.add_argument('--manifest',type=Path,default=Path(__file__).with_name('BUNDLE.json'))
ap.add_argument('--destination',type=Path,required=True)
a=ap.parse_args()
meta=json.loads(a.manifest.read_text())
with a.bundle.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==meta['bundle']['sha256'],'Bundle checksum mismatch'
assert a.bundle.stat().st_size==meta['bundle']['bytes']
assert not a.destination.exists(),'Choose a new destination; no saved output is overwritten'
records={r['path']:r for r in meta['members']}
with tarfile.open(a.bundle,'r:gz') as tf:
    members=tf.getmembers()
    assert len(members)==len(records) and {m.name for m in members}==set(records)
    for m in members:
        path=PurePosixPath(m.name)
        assert m.isfile() and not path.is_absolute() and '..' not in path.parts
        assert m.size==records[m.name]['bytes']
    a.destination.mkdir(parents=True)
    for m in members:
        target=a.destination/m.name;target.parent.mkdir(parents=True,exist_ok=True)
        with tf.extractfile(m) as src,target.open('xb') as out:shutil.copyfileobj(src,out)
        with target.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==records[m.name]['sha256'],m.name
print(json.dumps({'status':'PASS','files_verified':len(records),'destination':str(a.destination),'inference_run':False}))
