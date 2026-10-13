"""Verify and restore only this bounded research packet; standard library only."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath

def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--manifest',type=Path,default=Path(__file__).with_name('BUNDLE.json'));ap.add_argument('--destination',type=Path,required=True);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());assert a.bundle.stat().st_size==m['archive_bytes'] and digest(a.bundle)==m['archive_sha256'],'Archive checksum mismatch'
    expected={r['path']:r for r in m['files']};assert len(expected)==len(m['files'])
    with tarfile.open(a.bundle,'r:gz') as tf:
        members=tf.getmembers();assert len(members)==len(expected)
        seen=set()
        for member in members:
            p=PurePosixPath(member.name)
            assert member.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in member.name
            assert member.name in expected and member.name not in seen and member.size==expected[member.name]['bytes'];seen.add(member.name)
        a.destination.mkdir(parents=True,exist_ok=False)
        for member in members:
            with tf.extractfile(member) as f:body=f.read()
            assert len(body)==member.size and hashlib.sha256(body).hexdigest()==expected[member.name]['sha256'],member.name
            p=a.destination/member.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
    report={'status':'PASS','files':len(seen),'bytes':sum(r['bytes'] for r in expected.values()),'archive_sha256':m['archive_sha256'],'destination':str(a.destination.resolve()),'scope':'Checksummed file restoration only; no inference, acquisition or scientific replay.'}
    (a.destination/'RESTORE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
