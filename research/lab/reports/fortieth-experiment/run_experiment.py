"""Bounded, sequential one-arm coverage experiment against the exact retained control."""
import os,subprocess,sys,time
from common import *
def main():
    plan();out=HERE/'run-v1';out.mkdir(exist_ok=False);start=time.monotonic();deadline=start+1800;commands=[]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    jobs=[('teacher','run_teacher.py',['--deadline',str(deadline)]),('train-D_replication','run_training.py',['--deadline',str(deadline)]),('verify-training','verify_training.py',[]),('eval-D_replication-512','run_evaluation.py',['--arm','D_replication','--step','512','--deadline',str(deadline)]),('analyze','analyze.py',[])]
    try:
        for name,script,args in jobs:
            remaining=deadline-time.monotonic();require(remaining>0,'Combined wall limit');cmd=[sys.executable,'-B',str(HERE/script),*args];clock=time.monotonic();print(json.dumps(dict(event='START',job=name,remaining_seconds=remaining)),flush=True)
            with (out/(name+'.log')).open('xb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=remaining,cwd=HERE)
            commands.append(dict(job=name,command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-clock,log=pin(out/(name+'.log'))));save(out/(name+'-command.json'),commands[-1]);require(r.returncode==0,'Worker failed: '+name)
            require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<2*1024**3,'Output cap');print(json.dumps(dict(event='DONE',job=name,elapsed_seconds=time.monotonic()-start)),flush=True)
        require(time.monotonic()<deadline,'Final deadline');save(out/'COMPLETE.json',dict(status='PASS',created_at_utc=now(),GO=pin(HERE/'GO.json'),workers=commands,elapsed_seconds=time.monotonic()-start,optimizer_updates=512,training_forwards=705,teacher_forwards=32,development_forwards=200,network_requests=0,GPU_spend_GBP=0,retries=0))
    except Exception as e:
        save(out/'FAILURE.json',dict(status='STOP',error=type(e).__name__+': '+str(e),commands=commands,elapsed_seconds=time.monotonic()-start));raise
if __name__=='__main__':main()
