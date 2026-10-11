"""Offline checksum-verified restart restore; never overwrites different bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def restore(backup,destination):
    index=json.loads((backup/'INDEX.json').read_text())
    dest=destination.resolve();seen=set();restored=reused=0
    for g in index['generations']:
        archive=backup/g['name']
        assert archive.stat().st_size==g['bytes'] and sha(archive)==g['sha256'],'Missing LFS body or changed archive'
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers();assert len(members)==len(g['members'])
            for member,r in zip(members,g['members']):
                path=PurePosixPath(r['path'])
                assert member.isfile() and member.name==r['path'] and member.size==r['bytes']
                assert not path.is_absolute() and '..' not in path.parts and r['path'] not in seen
                seen.add(r['path']);target=dest/path
                assert target.resolve().is_relative_to(dest),'Symlink escapes destination'
                if target.exists():
                    assert target.is_file() and target.stat().st_size==r['bytes'] and sha(target)==r['sha256'],'Different existing file'
                    reused+=1;continue
                target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_name(target.name+'.restore-partial')
                with tar.extractfile(member) as src,tmp.open('wb') as out:
                    while block:=src.read(1024*1024):out.write(block)
                    out.flush();os.fsync(out.fileno())
                assert tmp.stat().st_size==r['bytes'] and sha(tmp)==r['sha256']
                os.replace(tmp,target);restored+=1
    assert sha(dest/'GO.json')==index['go_sha256']
    return dict(status='PASS',generations=len(index['generations']),files=len(seen),restored=restored,reused=reused,
                worker_started=False,source_prerequisites=index['prerequisites'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    print(json.dumps(restore(a.backup,a.destination),indent=2))
