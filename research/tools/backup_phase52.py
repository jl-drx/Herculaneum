"""Append verified CPU run inputs and committed batch deltas to a private Git ref.

Uses an independent Git tree/index-free commits; never stages or changes the
research checkout. Does not launch inference or use GitHub Actions. LFS payloads
are uploaded before a non-forced reference update. Previously published members
are immutable. Only a fixed study directory and receipt-owned outputs qualify.
"""
import argparse,datetime,fcntl,gzip,hashlib,io,json,os,subprocess,tarfile,time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
LAB=Path('/workspace/herculaneum-lab')
STUDY=LAB/'reports/fiftysecond-experiment'
REF='refs/heads/codex/plinius-pherc0800-phase52-checkpoints'
ROOT=Path('/workspace/scratch/phase52-backup')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args,data=None,env=None):
 p=subprocess.run(['git','-C',str(REPO),*args],input=data,capture_output=True,check=True,env=env)
 return p.stdout

def pin(p):return dict(path=str(p.relative_to(LAB)),bytes=p.stat().st_size,sha256=sha(p))
def blob(data):return git('hash-object','-w','--stdin',data=data).decode().strip()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,v):
 t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)

def snapshot():
 ROOT.mkdir(exist_ok=True);lock=(ROOT/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 go_path=STUDY/'INFERENCE_GO.json';go=json.loads(go_path.read_text());go_sha=sha(go_path)
 remote=git('ls-remote','origin',REF).decode().strip();parent=remote.split()[0] if remote else None
 if parent:
  git('fetch','--no-tags','origin',REF)
  prior=json.loads(git('show',parent+':INDEX.json'));assert prior['go_sha256']==go_sha
 else:prior=dict(schema_version=1,study='fiftysecond-experiment',go_sha256=go_sha,base_scientific_commit=git('rev-parse','HEAD').decode().strip(),generations=[],prerequisites=[r for r in go['pins']+go['runtime_pins']+[m['checkpoint'] for m in go['models']] if not Path(r['path']).is_relative_to(STUDY)])
 published={r['path']:r for g in prior['generations'] for r in g['members']}
 paths={p for p in STUDY.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json']}
 paths.update(p for p in (STUDY/'work/execution-qualification').rglob('*') if p.is_file())
 paths.add(STUDY/'work/execution-qualification.log')
 for r in go['pins']:
  p=Path(r['path'])
  if p.is_relative_to(STUDY):assert p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'];paths.add(p)
 for receipt in sorted((STUDY/'work/inference').glob('*/batch-*.json')):
  raw=receipt.read_bytes();r=json.loads(raw);assert r['binding']['go_sha256']==go_sha
  p=Path(r['logits']['path']);assert p.is_relative_to(STUDY/'work/inference') and p.stat().st_size==r['logits']['size_bytes'] and sha(p)==r['logits']['sha256'];paths.update([p,receipt])
 for receipt in sorted((STUDY/'work/inference').glob('*/COMPLETE.json')):
  r=json.loads(receipt.read_text());assert r['go_sha256']==go_sha;paths.add(receipt)
  for v in r['outputs'].values():
   p=Path(v['path']);assert p.is_relative_to(STUDY/'work/inference') and p.stat().st_size==v['size_bytes'] and sha(p)==v['sha256'];paths.add(p)
 if (STUDY/'INFERENCE_RESULTS.json').exists():paths.add(STUDY/'INFERENCE_RESULTS.json')
 members=[]
 for p in sorted(paths):
  assert p.is_file() and not p.is_symlink();r=pin(p)
  if r['path'] in published:assert published[r['path']]==r,r['path']
  else:members.append(r)
 if not members:
  atomic(ROOT/'STATUS.json',dict(status='UP_TO_DATE',utc=now(),remote_commit=parent,generations=len(prior['generations']),files=len(published)));return
 number=len(prior['generations']);name=f'generation-{number:04}.tar.gz';archive=ROOT/name
 # An unpublished incomplete archive may be safely regenerated; remote history
 # controls the generation number, never a local success assumption.
 with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
  for r in members:
   p=LAB/r['path'];assert pin(p)==r
   info=tar.gettarinfo(str(p),arcname=r['path']);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
   with p.open('rb') as f:tar.addfile(info,f)
   assert pin(p)==r,'Changed during backup'
 # Full member readback before any upload.
 with tarfile.open(archive,'r:gz') as tar:
  actual=tar.getmembers();assert [m.name for m in actual]==[r['path'] for r in members]
  for m,r in zip(actual,members):
   assert m.isfile() and m.size==r['bytes'];f=tar.extractfile(m);assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
 digest=sha(archive);size=archive.stat().st_size
 with archive.open('rb') as f:
  proc=subprocess.run(['git','-C',str(REPO),'lfs','clean','--',name],stdin=f,capture_output=True,check=True);pointer=proc.stdout
 assert f'oid sha256:{digest}\nsize {size}\n'.encode() in pointer
 git('lfs','push','--object-id','origin',digest)
 generation=dict(name=name,bytes=size,sha256=digest,members=members,created_at_utc=now(),archive_readback='PASS')
 prior['generations'].append(generation);prior['updated_at_utc']=now()
 entries={'.gitattributes':blob(b'*.tar.gz filter=lfs diff=lfs merge=lfs -text\n'),'INDEX.json':blob((json.dumps(prior,indent=2)+'\n').encode())}
 readme=b'''# Private PHerc0800 study52 restart backups\n\nAppend-only generations contain pinned inputs and committed raw model logits.\nThis is a partial running-study backup, not completed inference or a paper.\nRead INDEX.json for hashes, members, GO binding and earlier model/source prerequisites.\nFetch only the listed LFS objects. Restore via research/tools/restore_phase52_backup.py\nfrom the research branch, into /workspace/herculaneum-lab after checksum validation.\nThe pinned worker requires original absolute paths. Previous models/runtime are\nnot duplicated here; their identities are in prerequisites. No live process,\ninstalled environment or automatic platform restart is backed up.\n'''
 entries['README.md']=blob(readme)
 for g in prior['generations']:
  ptr=f"version https://git-lfs.github.com/spec/v1\noid sha256:{g['sha256']}\nsize {g['bytes']}\n".encode();entries[g['name']]=blob(ptr)
 tree=git('mktree',data=''.join(f'100644 blob {oid}\t{name}\n' for name,oid in sorted(entries.items())).encode()).decode().strip()
 args=['commit-tree',tree]
 if parent:args+=['-p',parent]
 commit=git(*args,data=f'Preserve PHerc0800 study52 checkpoint generation {number}\n'.encode()).decode().strip()
 # Non-force push rejects a concurrent writer. Failed publications stay retryable.
 git('push','origin',commit+':'+REF)
 actual=git('ls-remote','origin',REF).decode().split()[0];assert actual==commit
 atomic(ROOT/'STATUS.json',dict(status='PUSHED',utc=now(),remote_commit=commit,generations=len(prior['generations']),files=len(published)+len(members),new_files=len(members),archive_bytes=size,archive_sha256=digest,remote_ref=REF))
 print(json.dumps(json.loads((ROOT/'STATUS.json').read_text())),flush=True)

def alive(pid,start):
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();fields=raw.rsplit(')',1)[1].split();return fields[0]!='Z' and fields[19]==start
 except FileNotFoundError:return False

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch-pid',type=int);p.add_argument('--start-ticks');p.add_argument('--interval',type=int,default=600);a=p.parse_args()
 if not a.watch_pid:snapshot();return
 assert a.start_ticks and a.interval>=60
 while True:
  try:snapshot()
  except Exception as e:
   ROOT.mkdir(exist_ok=True);atomic(ROOT/'ERROR.json',dict(utc=now(),error=repr(e)));raise
  if not alive(a.watch_pid,a.start_ticks):return
  deadline=time.monotonic()+a.interval
  while time.monotonic()<deadline and alive(a.watch_pid,a.start_ticks):time.sleep(min(30,max(0,deadline-time.monotonic())))
if __name__=='__main__':main()
