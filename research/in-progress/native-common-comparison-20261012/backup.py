"""Append immutable restart snapshots to a dedicated PRIVATE Git/LFS branch.

Independent Git trees leave the research checkout/index untouched. No Actions.
Only checksum-committed logits and completed maps qualify.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from datetime import datetime, timezone

REF = 'refs/heads/codex/plinius-native-common-comparison-checkpoints'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    tmp=path.with_suffix('.partial')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def pin(p, root):
    return dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p))


def check(root,row):
    p=root/row['path']
    assert p.resolve().is_relative_to(root.resolve()) and p.is_file() and not p.is_symlink()
    assert pin(p,root)==row, str(p)
    return p


def collect(run):
    go=read(run/'GO.json');go_sha=sha(run/'GO.json')
    paths={run/'GO.json'}
    paths.update(check(run,r) for r in go['code_pins'])
    for folder in sorted((run/'jobs').glob('*')):
        if not folder.is_dir():continue
        for rp in sorted(folder.glob('tile-*.json')):
            row=read(rp);assert row['binding']['go_sha256']==go_sha
            paths.update([rp,check(folder,row['logits'])])
        complete=folder/'COMPLETE.json'
        if complete.exists():
            row=read(complete);assert row['binding']['go_sha256']==go_sha
            paths.add(complete);paths.update(check(folder,r) for r in row['outputs'])
    for name in ['INJECTED_STOP.json','INFERENCE_COMPLETE.json']:
        p=run/name
        if p.exists():
            assert read(p)['go_sha256']==go_sha;paths.add(p)
    result=run/'RESULTS.json'
    if result.exists():
        row=read(result);assert row['go_sha256']==go_sha
        paths.add(result);paths.update(check(run,r) for r in row['outputs'])
    return sorted(paths)


def snapshot(repo,run,state):
    state.mkdir(exist_ok=True,parents=True)
    lock=(state/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def git(*args,data=None):
        return subprocess.run(['git','-C',str(repo),*args],input=data,capture_output=True,check=True,timeout=180).stdout
    def blob(data):
        return git('hash-object','-w','--stdin',data=data).decode().strip()
    # Authenticated repository metadata, never enumerate secrets.
    metadata=json.loads(subprocess.run(['gh','repo','view','jl-drx/Herculaneum','--json','isPrivate,nameWithOwner'],
                         cwd=repo,capture_output=True,check=True,timeout=60).stdout)
    assert metadata=={'isPrivate':True,'nameWithOwner':'jl-drx/Herculaneum'}
    go=read(run/'GO.json');go_sha=sha(run/'GO.json')
    remote=git('ls-remote','origin',REF).decode().strip()
    parent=remote.split()[0] if remote else None
    if parent:
        git('fetch','--no-tags','origin',REF)
        index=json.loads(git('show',parent+':INDEX.json'))
        assert index['go_sha256']==go_sha
    else:
        index=dict(schema_version=1,study=go['plan']['study'],go_sha256=go_sha,generations=[],
                   base_scientific_commit=git('rev-parse','HEAD').decode().strip(),
                   prerequisites=go['source_archives'],note='Resume files, not a live process or installed runtime')
    published={r['path']:r for g in index['generations'] for r in g['members']}
    members=[]
    for p in collect(run):
        r=pin(p,run)
        if r['path'] in published:
            assert published[r['path']]==r,'Published input changed: '+r['path']
        else:
            members.append(r)
    if not members:
        atomic(state/'STATUS.json',dict(status='UP_TO_DATE',utc=now(),remote_commit=parent,
               remote_ref=REF,generations=len(index['generations']),files=len(published)))
        return
    number=len(index['generations']);name=f'generation-{number:04d}.tar.gz';archive=state/name
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for r in members:
            p=check(run,r)
            info=tar.gettarinfo(str(p),arcname=r['path']);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
            with p.open('rb') as f:tar.addfile(info,f)
            assert pin(p,run)==r
    with tarfile.open(archive,'r:gz') as tar:
        actual=tar.getmembers();assert [m.name for m in actual]==[r['path'] for r in members]
        for m,r in zip(actual,members):
            assert m.isfile() and m.size==r['bytes']
            with tar.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
    digest=sha(archive);size=archive.stat().st_size
    with archive.open('rb') as f:
        pointer=subprocess.run(['git','-C',str(repo),'lfs','clean','--',name],stdin=f,capture_output=True,check=True,timeout=120).stdout
    assert f'oid sha256:{digest}\nsize {size}\n'.encode() in pointer
    git('lfs','push','--object-id','origin',digest)
    index['generations'].append(dict(name=name,bytes=size,sha256=digest,members=members,created_utc=now(),archive_readback='PASS'))
    index['latest_worker_status']=read(run/'STATUS.json') if (run/'STATUS.json').exists() else None
    index['updated_utc']=now()
    entries={'.gitattributes':blob(b'*.tar.gz filter=lfs diff=lfs merge=lfs -text\n'),
             'INDEX.json':blob((json.dumps(index,indent=2)+'\n').encode()),
             'README.md':blob(b'# Private native common-support comparison restart backups\n\nAppend-only verified generations. INDEX.json lists checksums, GO identity and\nseparate source archives. Fetch these LFS objects and use restore_backup.py from the native0139 packet on\nthe research branch to restore into a fresh run directory. See its README.md for\ncommands. This preserves committed inference progress, not a running process,\ninstalled runtime or automatic platform restart. No GPU or Actions job is used.\n')}
    for g in index['generations']:
        pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{g['sha256']}\nsize {g['bytes']}\n".encode()
        entries[g['name']]=blob(pointer)
    tree=git('mktree',data=''.join(f'100644 blob {oid}\t{name}\n' for name,oid in sorted(entries.items())).encode()).decode().strip()
    args=['commit-tree',tree]
    if parent:args+=['-p',parent]
    commit=git(*args,data=f'Preserve native model comparison generation {number}\n'.encode()).decode().strip()
    git('push','origin',commit+':'+REF)
    assert git('ls-remote','origin',REF).decode().split()[0]==commit
    result=dict(status='PUSHED',utc=now(),remote_commit=commit,remote_ref=REF,generations=len(index['generations']),
                files=len(published)+len(members),new_files=len(members),archive_bytes=size,archive_sha256=digest)
    atomic(state/'STATUS.json',result);print(json.dumps(result),flush=True)


def alive(worker):
    try:
        fields=Path(f"/proc/{worker['pid']}/stat").read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and fields[19]==worker['start_ticks']
    except FileNotFoundError:
        return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args();worker=read(a.run/'WORKER.json') if a.watch else None
    interval=read(a.run/'GO.json')['plan']['backup_interval_seconds']
    while True:
        try:
            snapshot(a.repo,a.run,a.state)
        except Exception as e:
            a.state.mkdir(parents=True,exist_ok=True)
            atomic(a.state/'ERROR.json',dict(utc=now(),error_type=type(e).__name__,message=str(e)))
            # Training continues; make failure visible and retry on the next
            # interval. Do not erase the last successful backup receipt.
            if not a.watch:raise
        if not a.watch or not alive(worker):return
        deadline=time.monotonic()+interval
        while time.monotonic()<deadline and alive(worker):
            time.sleep(min(20,max(0,deadline-time.monotonic())))


if __name__=='__main__':main()
