"""Private portable checkpoint of a partial, explicitly unadmitted population."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import shutil
from intake import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    if a.archive.exists() or a.receipt.exists():raise FileExistsError('Use new output paths')
    sources={}
    def add(source,dest):
        assert source.is_file() and not source.is_symlink() and dest not in sources
        sources[dest]=source
    for source in sorted(a.work.rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            assert source.suffix!='.partial','Unfinished write requires explicit review'
            add(source,'work/'+str(source.relative_to(a.work)))
    plan=json.loads((a.work/'REQUEST_PLAN.json').read_text())
    for name,digest in plan['base_pins'].items():
        assert sha(a.base/name)==digest;add(a.base/name,'base/'+name)
    packet=a.repo/'research/in-progress/native-coverage-admission-20261012'
    for source in sorted(packet.iterdir()):
        if source.suffix=='.py' or source.name in ['PROTOCOL.md','README.md']:
            add(source,'code/'+source.name)
    for path in [
        a.repo/'research/in-progress/native-common-support-20261012/verify_bundle.py',
        a.repo/'research/planning/pherc0139-learning-readiness-20261011/native-reference/intake.py']:
        add(path,'dependencies/'+('verify_bundle.py' if path.name=='verify_bundle.py' else 'retained_native_annotation_intake.py'))
    with tempfile.TemporaryDirectory(prefix='plinius-admission-package-',dir='/tmp') as folder:
        root=Path(folder)
        for name,source in sources.items():
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            assert sha(source)==sha(dest)
        (root/'README.md').write_text(
            '# Private partial population checkpoint\n\n'
            'Read code/README.md. This is not a complete or admitted training population.\n'
            'The public S3 input transport failed with proxy-tunnel HTTP 503 errors.\n'
            'Every completed source object, request/receipt, annotation shard, decoder\n'
            'result and offline qualification is included. Unknown input is not blank.\n\n'
            'Verify the external archive SHA and all manifest entries with dependencies/verify_bundle.py.\n'
            'Then run code/verify_original.py --work /path/to/work --output /new/check.json.\n'
            'This independently reproduces all 876 native candidate decisions and 481\n'
            'complete CT-supported rows using NumPy, no network or neural inference.\n\n'
            'code/continue_acquisition.py inspect --base /path/to/base --work /path/to/work\n'
            'checks the cache without network. Repeat for --work /path/to/work/original.\n'
            'Only after S3 transport is restored, acquire with the same paths and a new\n'
            '--attempt identifier. A probe must pass before bulk reads; data requests,\n'
            'eligibility and saved receipts are unchanged. Completed historical\n'
            'qualification outputs must not be overwritten or treated as a new result.\n\n'
            'No models, environments, credentials, active processes or fresh evaluation\n'
            'payloads are included. Older absolute reuse paths are historical provenance;\n'
            'all referenced reusable objects have already been materialised in this cache.\n'
            'The source catalogue/URLs attribute Vesuvius Challenge public data. Retain\n'
            'its CC BY-NC 4.0 metadata and separate code/data provenance. Keep private.\n')
        files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()}
        (root/'MANIFEST.json').write_text(json.dumps(dict(schema=1,scope='Partial research input checkpoint, not full cohort admission',files=files),indent=2)+'\n')
        (root/'SHA256SUMS').write_text(''.join(f"{v['sha256']}  {k}\n" for k,v in files.items()))
        a.archive.parent.mkdir(parents=True,exist_ok=True)
        with a.archive.open('xb') as target:
            with gzip.GzipFile(filename='',fileobj=target,mode='wb',compresslevel=6,mtime=0) as compressed:
                with tarfile.open(fileobj=compressed,mode='w') as tar:
                    for source in sorted(root.rglob('*')):
                        if not source.is_file():continue
                        info=tar.gettarinfo(str(source),arcname=str(source.relative_to(root)))
                        info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o644
                        with source.open('rb') as stream:tar.addfile(info,stream)
        receipt=dict(status='PACKAGED_PENDING_RESTORE',path=str(a.archive.relative_to(a.repo)),
            bytes=a.archive.stat().st_size,sha256=sha(a.archive),manifest_files=len(files),
            payload_bytes=sum(r['bytes'] for r in files.values()),manifest_sha256=sha(root/'MANIFEST.json'),
            implementation_sha256=sha(Path(__file__)),full_population_admitted=False,model_weights_included=False,paid_compute=False)
        a.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
