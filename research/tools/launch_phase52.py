"""Start the qualified fixed CPU run and checkpoint watcher; never rent compute.

Run only after the explicit qualification and remote-restoration gates. Keeps
previous launch records and refuses duplicate live workers. No automatic restart
is installed; a new invocation after interruption must be deliberate.
"""
import datetime,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
P=Path('/workspace/herculaneum-lab/reports/fiftysecond-experiment');W=P/'work'
PYTHON=Path('/workspace/.venvs/pherc343/bin/python')
def alive(pid,start,script):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return fields[0]!='Z' and fields[19]==start and str(script).encode() in args
    except FileNotFoundError:return False
def atomic(path,value):
    tmp=path.with_suffix('.partial')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def main():
    lock=(W/'LAUNCH.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=W/'RUN_STATE.json';worker=P/'worker.py'
    if state.exists():
        old=json.loads(state.read_text())
        if alive(old['pid'],old['start_ticks'],worker):raise SystemExit('Worker already running')
        if old.get('watcher') and alive(old['watcher']['pid'],old['watcher']['start_ticks'],REPO/'research/tools/backup_phase52.py'):raise SystemExit('Previous checkpoint watcher still running')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert str(worker).encode() not in args,'Unrecorded duplicate worker'
    assert not (P/'INFERENCE_RESULTS.json').exists(),'Study already completed'
    go=json.loads((P/'INFERENCE_GO.json').read_text());digest=hashlib.sha256((P/'INFERENCE_GO.json').read_bytes()).hexdigest()
    assert go['status']=='FROZEN_FOR_SHIFTED_PRODUCTION' and go['expected_batches']==3096
    assert json.loads((P/'BATCH_QUALIFICATION.json').read_text())['status']=='PASS'
    proof=json.loads((P/'LAUNCH_BACKUP_VALIDATION.json').read_text())
    assert proof['status']=='PASS' and proof['go_sha256']==digest and proof['fresh_remote_download'] and proof['restore_pass']
    tmp=Path('/workspace/scratch/phase52-tmp');tmp.mkdir(exist_ok=True)
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TMPDIR=str(tmp))
    if state.exists():
        history=W/'launch-history';history.mkdir(exist_ok=True)
        dest=history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');assert not dest.exists();dest.write_bytes(state.read_bytes())
    with (W/'inference-worker.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([str(PYTHON),'-B',str(worker)],cwd=P,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    record=dict(pid=child.pid,start_ticks=start,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),go_sha256=digest,expected_maps=32,expected_batches=3096,precision='fp32 CPU',threads=4,gpu_spend_GBP=0)
    atomic(state,record)
    backup=Path('/workspace/scratch/phase52-backup');backup.mkdir(exist_ok=True)
    with (backup/'watch.log').open('ab',buffering=0) as log:
        watcher=subprocess.Popen([sys.executable,'-B',str(REPO/'research/tools/backup_phase52.py'),'--watch-pid',str(child.pid),'--start-ticks',start,'--interval','600'],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ticks=Path(f'/proc/{watcher.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    record['watcher']=dict(pid=watcher.pid,start_ticks=ticks,interval_seconds=600)
    atomic(state,record);atomic(backup/'WATCH_STATE.json',record['watcher'])
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
