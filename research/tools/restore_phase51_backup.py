"""Verify a downloaded phase51 checkpoint branch and restore without overwriting.

No network, credentials or inference. Requires each named LFS object to be fully
materialized alongside INDEX.json. Models/runtime prerequisites remain separate.
"""
import argparse,hashlib,json,tarfile,os
from pathlib import Path,PurePosixPath

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backup',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();index=json.loads((a.backup/'INDEX.json').read_text());dest=a.destination.resolve();seen=set();restored=0;reused=0
 for g in index['generations']:
  archive=a.backup/g['name'];assert archive.stat().st_size==g['bytes'] and sha(archive)==g['sha256'],'LFS pointer or changed archive'
  with tarfile.open(archive,'r:gz') as tar:
   members=tar.getmembers();assert len(members)==len(g['members'])
   for member,r in zip(members,g['members']):
    path=PurePosixPath(r['path']);assert member.isfile() and member.name==r['path'] and member.size==r['bytes'];assert path.parts[:2]==('reports','fiftyfirst-experiment') and not path.is_absolute() and '..' not in path.parts and r['path'] not in seen;seen.add(r['path']);target=dest/path
    assert target.resolve().is_relative_to(dest),'Symlink escapes destination'
    if target.exists():assert target.is_file() and target.stat().st_size==r['bytes'] and sha(target)==r['sha256'],'Existing file differs';reused+=1;continue
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.restore-partial')
    with tar.extractfile(member) as src,tmp.open('wb') as out:
     while block:=src.read(1024*1024):out.write(block)
     out.flush();os.fsync(out.fileno())
    assert tmp.stat().st_size==r['bytes'] and sha(tmp)==r['sha256'];os.replace(tmp,target);restored+=1
 assert sha(dest/'reports/fiftyfirst-experiment/INFERENCE_GO.json')==index['go_sha256']
 print(json.dumps(dict(status='PASS',generations=len(index['generations']),files=len(seen),restored=restored,reused=reused,inference_run=False,prerequisite_files_not_included=len(index['prerequisites']),scope='Restored pinned study inputs and committed outputs only; full worker startup still requires prior model/runtime files and the qualified environment.')))
if __name__=='__main__':main()
