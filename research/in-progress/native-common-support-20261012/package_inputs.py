"""Build a deterministic private bundle; deduplicate identical files by hardlink."""
import argparse,gzip,hashlib,io,json,tarfile
from pathlib import Path
import intake


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args()
    files={}
    for panel in ['pilot','supervision-pilot']:
        for p in sorted((a.base/panel).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            assert p.suffix!='.partial','Uncommitted body in packet'
            files[str(p.relative_to(a.base))]=p
    here=Path(__file__).parent
    for name in ['README.md','requirements-inputs.txt','COMPUTE_PLAN.json','MODEL_PINS.json','NETWORK_ACCOUNTING.json',
                 'SAMPLING_CHECKS.json','verify_bundle.py','package_inputs.py']:
        files['overview/'+name]=here/name
    records={n:intake.pin(p) for n,p in sorted(files.items())}
    manifest=json.dumps({'format':1,'scope':'Two development input panels; no model weights or inference results','files':records},indent=2)+'\n'
    sums=''.join(f"{r['sha256']}  {name}\n" for name,r in records.items())
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    if a.archive.exists():raise FileExistsError(a.archive)
    dedup={};links=0
    with a.archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as gz,tarfile.open(fileobj=gz,mode='w|') as tf:
        for name,p in sorted(files.items()):
            rec=records[name];key=(rec['bytes'],rec['sha256']);info=tarfile.TarInfo(name);info.mode=0o644;info.mtime=0
            if key in dedup:
                info.type=tarfile.LNKTYPE;info.linkname=dedup[key];tf.addfile(info);links+=1
            else:
                info.size=rec['bytes'];dedup[key]=name
                with p.open('rb') as stream:tf.addfile(info,stream)
        for name,body in [('MANIFEST.json',manifest.encode()),('SHA256SUMS',sums.encode())]:
            info=tarfile.TarInfo(name);info.mode=0o644;info.mtime=0;info.size=len(body);tf.addfile(info,io.BytesIO(body))
    result={'status':'PACKAGED_NOT_YET_RESTORED','archive':a.archive.name,**intake.pin(a.archive),'files':len(files),
            'internal_deduplication_hardlinks':links,'logical_payload_bytes':sum(r['bytes'] for r in records.values()),
            'manifest_sha256':hashlib.sha256(manifest.encode()).hexdigest(),'model_weights_included':False,
            'new_inference':0,'private_only':True}
    intake.write(a.receipt,result);print(json.dumps(result))


if __name__=='__main__':main()
