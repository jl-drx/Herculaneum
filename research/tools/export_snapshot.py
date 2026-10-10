"""Explicit text-only research export. Preserve original bytes and record every copy."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def select(root):
    chosen=set()
    for p in root.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if any(x in rel.parts for x in ['__pycache__','.git','.private','tools','plot-cache','cache','portability-phase39']):continue
        if any(s in str(rel).lower() for s in ['runpod-setup','credential','secret','oauth','setup-validation','r2-readiness']):continue
        if p.name.startswith('STARTUP') or p.name=='final-verification.json':continue
        if 'startup-save-receipt' in p.name or p.name.startswith('fontlist-'):continue
        if p.suffix in {'.py','.md','.sh','.toml','.yaml','.yml'} or p.name in {'requirements.lock','requirements.pinned.txt'}:chosen.add(p)
        elif p.suffix=='.json' and p.stat().st_size<=500_000 and (len(rel.parts)<=3 or ('thirtyninth-experiment' in rel.parts and len(rel.parts)<=4)):chosen.add(p)
        elif p.suffix=='.csv' and p.stat().st_size<1_000_000:chosen.add(p)
        elif any(s in p.name.upper() for s in ['LICENSE','NOTICE','COPYING']) and p.stat().st_size<500_000:chosen.add(p)
    # Include all nonbinary current replay dependencies regardless of nesting.
    phase=root/'reports/thirtyninth-experiment'
    for item in json.loads((phase/'GO.json').read_text())['pins']:
        path=Path(item['path'])
        if path.suffix in {'.py','.md','.json'}:chosen.add(path)
    chosen.add(phase/'comparison.png')
    return sorted(chosen)

def scan(paths):
    # Report locations/rules only, never credential-like matches.
    rules={
        'GitHub token':r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
        'AWS access key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        'Private key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'Authenticated URL':r'https?://[^\s/:]+:[^\s/@]+@',
        'Signed URL':r'[?&](?:X-Amz-Signature|X-Amz-Security-Token|access_token)=',
    }
    failures=[]
    for path in paths:
        if path.suffix=='.png':continue
        text=path.read_text(encoding='utf-8')
        for name,expression in rules.items():
            for match in re.finditer(expression,text,re.IGNORECASE if name=='Signed URL' else 0):
                if (name=='Authenticated URL' and path.name=='check_fixtures.py'
                        and 'twentysecond-experiment/benchmark-metadata/' in path.as_posix()
                        and match.group() in {'https://user:pass@','https://u:p@','https://user:password@'}):
                    continue  # Reviewed synthetic URL-rejection fixtures, no real credential.
                failures.append(dict(path=str(path),line=text.count('\n',0,match.start())+1,rule=name))
    if failures:raise SystemExit(json.dumps(dict(status='REVIEW_REQUIRED',locations=failures)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();root=args.lab.resolve();out=args.destination.resolve()
    paths=select(root);scan(paths);out.mkdir(parents=True,exist_ok=False);records=[]
    for source in paths:
        relative=source.relative_to(root);target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        checksum=digest(source)
        if digest(target)!=checksum:raise SystemExit('Copy identity failure: '+str(relative))
        records.append(dict(path=relative.as_posix(),source=str(source),sha256=checksum,size_bytes=source.stat().st_size))
    manifest=dict(schema_version=1,scope='Selected original research text/source plus phase39 figure; excludes bulk data, checkpoints, caches, startup histories and credentials',record_count=len(records),bytes=sum(r['size_bytes'] for r in records),records=records)
    (out.parent/'GIT_EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(files=len(records),bytes=manifest['bytes'],credential_pattern_scan='PASS',all_copies_byte_identical=True)))

if __name__=='__main__':main()
