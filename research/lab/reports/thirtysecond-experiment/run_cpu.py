"""External wall/RSS supervisor for one CPU worker; no automatic retries."""
import os,signal,subprocess,time
from common import *

def main():
    require(not (HERE/'CPU_SUPERVISOR.json').exists(),'CPU job already attempted')
    check_freeze('MODEL_GO.json')
    started=time.monotonic();receipt=dict(status='STARTING',started_utc=now(),model_GO=pin(HERE/'MODEL_GO.json'))
    save(HERE/'CPU_SUPERVISOR.json',receipt)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    reason=None;peak=0
    with (HERE/'CPU_WORKER.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-B',str(HERE/'model_job.py'),'worker'],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                if time.monotonic()-started>600:reason='External600second deadline';break
                try:
                    fields=Path(f'/proc/{child.pid}/status').read_text().splitlines()
                    rss=max((int(line.split()[1])*1024 for line in fields if line.startswith(('VmRSS:','VmHWM:'))),default=0)
                    peak=max(peak,rss)
                    if peak>4*1024**3:reason='External4GiB RSS cap';break
                except FileNotFoundError:pass
                time.sleep(.5)
            if reason:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            returncode=child.wait()
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
    receipt.update(status='PASS' if returncode==0 and reason is None else 'STOP',exit_code=returncode,stop_reason=reason,
                   elapsed_seconds=time.monotonic()-started,observed_peak_RSS_bytes=peak,completed_utc=now(),log=pin(HERE/'CPU_WORKER.log'))
    if receipt['status']=='PASS':
        result=json.loads((HERE/'inference-v1/COMPLETE.json').read_text())
        require(result['completed_forwards']==20 and result['model_state_unchanged'],'Worker completion invariant')
        receipt['completion']=pin(HERE/'inference-v1/COMPLETE.json')
    save(HERE/'CPU_SUPERVISOR.json',receipt);print(json.dumps(receipt,indent=2))
    require(receipt['status']=='PASS','CPU job stopped; do not retry automatically')

if __name__=='__main__':main()
