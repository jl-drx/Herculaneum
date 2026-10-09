"""Add one sealed study's selected evidence without changing archived originals."""
import argparse,hashlib,json,shutil
from pathlib import Path
from export_snapshot import scan

def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab',type=Path,required=True)
    parser.add_argument('--study',required=True)
    parser.add_argument('--register',required=True)
    args=parser.parse_args();lab=args.lab.resolve();root=Path(__file__).resolve().parents[1];phase=lab/'reports'/args.study
    if not (phase/'final-verification.json').exists():raise SystemExit('Study must be sealed before export')
    source_manifest=json.loads((phase/'final-verification.json').read_text());frozen={x['path']:x for x in source_manifest['records']}
    candidates=[]
    for p in phase.rglob('*'):
        if not p.is_file() or any(k in p.parts for k in ['__pycache__','plot-cache']):continue
        if p.name=='final-verification.json' or p.name.startswith('STARTUP'):continue
        if p.suffix in {'.py','.md','.json','.jsonl','.csv','.png'} and p.stat().st_size<=1_000_000:candidates.append(p)
    candidates.append(lab/'reports'/args.register);scan([p for p in candidates if p.suffix!='.png'])
    target_manifest=root/'GIT_EXPORT_MANIFEST.json';old_bytes=target_manifest.read_bytes();manifest=json.loads(old_bytes);records={r['path']:r for r in manifest['records']};previous=dict(records)
    for p in sorted(candidates):
        relative=p.relative_to(lab).as_posix();checksum=digest(p);item=dict(path=relative,source=str(p),sha256=checksum,size_bytes=p.stat().st_size)
        expected=frozen[str(p)]
        if (checksum,p.stat().st_size)!=(expected['sha256'],expected['size_bytes']):raise SystemExit('Sealed source changed')
        if relative in records and records[relative]!=item:raise SystemExit('Archived record collision: '+relative)
        target=root/'lab'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=checksum:raise SystemExit('Existing archive differs: '+relative)
        else:shutil.copyfile(p,target)
        records[relative]=item
    if any(records[k]!=v for k,v in previous.items()):raise SystemExit('Historical export changed')
    for r in records.values():
        p=root/'lab'/r['path']
        if digest(p)!=r['sha256'] or p.stat().st_size!=r['size_bytes']:raise SystemExit('Export identity: '+r['path'])
    ordered=[records[k] for k in sorted(records)]
    manifest.update(scope='Selected original research source/evidence and figures through '+args.study+'; excludes bulk arrays/checkpoints and full retention manifests',record_count=len(ordered),bytes=sum(r['size_bytes'] for r in ordered),records=ordered,previous_export_manifest_sha256=hashlib.sha256(old_bytes).hexdigest(),previous_records_preserved=len(previous))
    target_manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='PASS',added_records=len(records)-len(previous),previous_records_preserved=len(previous),records=len(records),bytes=manifest['bytes'],all_original_copies_exact=True)))
if __name__=='__main__':main()
