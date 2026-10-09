"""Verify and replay phase39 in a new directory using a separately downloaded capsule."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();descriptor=json.loads((Path(__file__).resolve().parents[1]/'CAPSULE.json').read_text())
    if digest(args.capsule)!=descriptor['sha256'] or args.capsule.stat().st_size!=descriptor['size_bytes']:raise SystemExit('Capsule digest/length mismatch')
    out=args.destination.resolve();out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(args.capsule,'r:gz') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        if len(set(names))!=len(names):raise SystemExit('Duplicate member')
        for m in members:
            name=PurePosixPath(m.name)
            if not m.isfile() or name.is_absolute() or '..' in name.parts or '\\' in m.name:raise SystemExit('Unsafe capsule member')
        manifest=json.load(archive.extractfile('CAPSULE_MANIFEST.json'))
        if set(names)!={r['path'] for r in manifest['records']}|{'CAPSULE_MANIFEST.json'}:raise SystemExit('Unexpected capsule files')
        archive.extractall(out,filter='data')
    for r in manifest['records']:
        p=out/r['path']
        if digest(p)!=r['sha256'] or p.stat().st_size!=r['size_bytes']:raise SystemExit('Restored input mismatch: '+r['path'])
    lab=out/'lab';phase=lab/'reports/thirtyninth-experiment';result=out/'replay-result'
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    subprocess.run([sys.executable,'-B',str(phase/'run_comparison.py'),'--lab-root',str(lab),'--output',str(result)],env=env,check=True,timeout=120)
    subprocess.run([sys.executable,'-B',str(phase/'verify_comparison.py'),'--lab-root',str(lab),'--run',str(result)],env=env,check=True,timeout=120)
    expected=json.loads((phase/'run-v1/METRICS.json').read_text());actual=json.loads((result/'METRICS.json').read_text())
    for key in ['roles','region_deltas_vs_seed42','worst_deltas_vs_seed42','advancement_screen','descriptive_relation']:
        if actual[key]!=expected[key]:raise SystemExit('Replay differs from retained '+key)
    proof=dict(status='PASS',capsule_sha256=descriptor['sha256'],restored_files=len(manifest['records']),all_reported_metrics_and_decisions_identical=True,scope='Relocated local replay using this interpreter; not an independent machine validation',python=sys.version,output=str(result))
    (out/'REPLAY_RECEIPT.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))

if __name__=='__main__':main()
