"""Create a small private replay artifact, never scan data or model checkpoints."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--descriptor',type=Path,required=True)
    args=parser.parse_args();lab=args.lab.resolve();phase=lab/'reports/thirtyninth-experiment'
    go=json.loads((phase/'GO.json').read_text());paths=set()
    for p in go['pins']:
        source=lab/Path(p['path']).relative_to('/workspace/herculaneum-lab')
        if digest(source)!=p['sha256'] or source.stat().st_size!=p['size_bytes']:raise SystemExit('Input hash mismatch')
        paths.add(source)
    for name in ['GO.json','RESULTS.md','run-v1/METRICS.json','run-v1/COMPLETE.json','run-v1/VERIFICATION.json']:paths.add(phase/name)
    records=[]
    for p in sorted(paths):
        if p.suffix not in {'.py','.md','.json','.npy'}:raise SystemExit('Unexpected capsule type')
        records.append(dict(path='lab/'+p.relative_to(lab).as_posix(),sha256=digest(p),size_bytes=p.stat().st_size))
    manifest=dict(schema_version=1,records=records,scope='Only phase39 frozen inputs and reference evidence; no raw CT scans or model weights')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as archive:
        for source,record in zip(sorted(paths),records):
            info=tarfile.TarInfo(record['path']);info.size=record['size_bytes'];info.mode=0o644
            with source.open('rb') as stream:archive.addfile(info,stream)
        payload=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();info=tarfile.TarInfo('CAPSULE_MANIFEST.json');info.size=len(payload);info.mode=0o644;archive.addfile(info,io.BytesIO(payload))
    descriptor=dict(schema_version=1,file=args.output.name,sha256=digest(args.output),size_bytes=args.output.stat().st_size,files=len(records),uncompressed_payload_bytes=sum(r['size_bytes'] for r in records),release_tag='plinius-phase39-replay',repository='jl-drx/Herculaneum',release_intent='private repository draft asset; metadata verified after upload',python='3.12',numpy='2.5.3',scope=manifest['scope'])
    with args.descriptor.open('x') as stream:json.dump(descriptor,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(descriptor))

if __name__=='__main__':main()
