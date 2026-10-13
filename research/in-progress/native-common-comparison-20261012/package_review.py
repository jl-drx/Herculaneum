"""Create a private, self-contained numeric review bundle after verification.

Includes committed logits/maps, scoring references, definitions and verification
code. Excludes model weights and CT source volumes. No inference or network.
"""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import runner as r


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    assert not a.archive.exists() and not a.receipt.exists()
    done=r.read(a.run/'RESULTS.json');go=r.read(a.run/'GO.json')
    assert done['status']=='VERIFIED_DEVELOPMENT_COMPARISON_COMPLETE' and done['go_sha256']==r.sha(a.run/'GO.json')
    for row in done['outputs']:r.check(a.run,row)
    spec=importlib.util.spec_from_file_location('frozen_backup',a.run/'implementation/backup.py')
    backup=importlib.util.module_from_spec(spec);spec.loader.exec_module(backup)
    members={str(Path('run')/path.relative_to(a.run)):path for path in backup.collect(a.run)}
    for row in go['input_pins']:
        path=r.check(a.inputs,row);members[str(Path('inputs')/row['path'])]=path
    description='''# Native common-support numeric review bundle

Private development evidence, 12 October 2026. Five released models, four
segments on two exposed papyri, 24 fixed cores, 120 maps and 1,698 saved logit
tiles. The references are transferred/model-reviewed, with unresolved upstream
exposure and physical registration. They are not authenticated text or ink truth.

Included: immutable inference receipts/logits/maps, scoring reference arrays,
all metric tables, fixed-scale panels, protocols, source/checkpoint identities,
qualified verifier and code. MANIFEST.json and SHA256SUMS cover each payload.
Excluded: model weights, full CT or source annotation arrays, all 1,050 prepared
CT input patches, credentials, installed environments and live processes. The
full separately backed-up input bundle remains necessary to recheck CT/input
construction or run inference. Retained input receipts are not a new source audit.

For numeric review, install numpy==2.4.4 and opencv-python-headless==5.0.0.93 in an
isolated Python environment, then from this extracted directory run:

    python run/evaluation-implementation/verify_results.py --inputs inputs --run run --output my-verification.json

This verifies file and decoded-logit hashes, receipt/model/input bindings, all
120 maps, their declared quantization/resizing, 238 metric records and comparison
flags. It reads saved logits; it never runs a model or needs torch, CUDA or keys.
This is a separate implementation, not an external human review. For fresh human
interpretation, all four six-core PNG panels use fixed display scaling; inspect
every core, including all 14 one-class cores. Do not select only attractive maps.

The original producer already replayed every job from logits exactly. This
standalone verifier checks the saved replay receipts and independently recomputes
map derivatives and metrics; it does not itself rerun the original accumulation.

See run/evaluation/TABLES.md and METRICS.json for results. The same underlying
physical material is not a new held-out dataset on another computer.
'''
    generated={'README.md':description.encode()}
    manifest={'schema_version':1,'go_sha256':done['go_sha256'],'scope':'Saved-output numeric review; source CT and model weights excluded',
              'files':{name:{'bytes':path.stat().st_size,'sha256':r.sha(path)} for name,path in sorted(members.items())}}
    for name,data in generated.items():manifest['files'][name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    manifest['files']=dict(sorted(manifest['files'].items()))
    generated['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    generated['SHA256SUMS']=''.join(f'{row["sha256"]}  {name}\n' for name,row in manifest['files'].items()).encode()
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    with a.archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for name in sorted(set(members)|set(generated)):
            if name in generated:
                data=generated[name];info=tarfile.TarInfo(name);info.size=len(data)
                info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(data))
            else:
                path=members[name];info=tar.gettarinfo(str(path),arcname=name)
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with path.open('rb') as f:tar.addfile(info,f)
                assert r.sha(path)==manifest['files'][name]['sha256']
    with tarfile.open(a.archive,'r:gz') as tar:
        names=tar.getnames();assert len(names)==len(set(names))==len(manifest['files'])+2
        for name,row in manifest['files'].items():
            member=tar.getmember(name);assert member.isfile() and member.size==row['bytes']
            with tar.extractfile(member) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    r.atomic(a.receipt,dict(status='PACKAGED_AND_ARCHIVE_READBACK_VERIFIED',archive=r.pin(a.archive),
             files=len(manifest['files']),go_sha256=done['go_sha256'],trained_model_forwards=0,
             requires_source_bundle_for_input_audit=True,private_only=True,implementation=r.pin(Path(__file__))))
    print(json.dumps(r.read(a.receipt),indent=2))


if __name__=='__main__':main()
