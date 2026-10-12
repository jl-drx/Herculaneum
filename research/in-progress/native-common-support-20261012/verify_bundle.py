"""Extract a private inputs bundle and verify every manifest entry, offline."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);a=p.parse_args()
    assert sha(a.archive)==a.expected_sha256,'Archive SHA256 mismatch'
    if a.output.exists():raise FileExistsError('Use a new extraction directory')
    a.output.mkdir(parents=True)
    with tarfile.open(a.archive,'r:gz') as tf:
        members=tf.getmembers();seen=set()
        for m in members:
            path=PurePosixPath(m.name)
            assert not path.is_absolute() and '..' not in path.parts and m.name not in seen
            assert m.isfile() or m.islnk(),'Only regular files and internal deduplication hardlinks allowed'
            if m.islnk():assert m.linkname in seen,'Hardlink must name an earlier archive member'
            seen.add(m.name)
        tf.extractall(a.output,filter='data')
    manifest=json.loads((a.output/'MANIFEST.json').read_text())
    expected=set(manifest['files'])|{'MANIFEST.json','SHA256SUMS'}
    actual={str(p.relative_to(a.output)) for p in a.output.rglob('*') if p.is_file()}
    assert actual==expected,'Missing or unexpected extracted files'
    for name,r in manifest['files'].items():
        f=a.output/name;assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],name
    sums=''.join(f"{r['sha256']}  {name}\n" for name,r in manifest['files'].items())
    assert (a.output/'SHA256SUMS').read_text()==sums
    print(json.dumps({'status':'PASS','files_verified':len(manifest['files']),
                      'archive_sha256':a.expected_sha256,'new_inference':0}))


if __name__=='__main__':main()
