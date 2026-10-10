"""Sequential workers with independent 30-minute controller deadline; no retries."""
import os,subprocess,sys,time
from study_common import *
def main():
    p=plan();out=HERE/'run-v1';out.mkdir(exist_ok=False);start=time.monotonic();deadline=start+1800
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    commands=[]
    jobs=[('train-'+a,'run_training.py',['--arm',a,'--deadline',str(deadline)]) for a in ARMS]
    jobs+=[('verify-training','verify_training.py',[])]
    jobs += [(f'eval-{a}-{n}','run_evaluation.py',['--arm',a,'--step',str(n),'--deadline',str(deadline)]) for a in ARMS for n in [32,512]]
    jobs += [('analyze','analyze.py',[])]
    try:
        for name,script,args in jobs:
            remaining=deadline-time.monotonic();require(remaining>0,'Combined30minute limit')
            cmd=[sys.executable,'-B',str(HERE/script),*args];clock=time.monotonic();print(json.dumps(dict(event='START',job=name,remaining_seconds=remaining)),flush=True)
            with (out/(name+'.log')).open('xb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=remaining,cwd=HERE)
            commands.append(dict(job=name,command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-clock,log=pin(out/(name+'.log'))))
            save(out/(name+'-command.json'),commands[-1]);require(r.returncode==0,'Worker failed: '+name)
            require(sum(x.stat().st_size for x in HERE.rglob('*') if x.is_file())<2*1024**3,'Combined2GiB output limit')
            print(json.dumps(dict(event='DONE',job=name,elapsed_seconds=time.monotonic()-start)),flush=True)
        require(time.monotonic()<deadline,'Final combined deadline')
        save(out/'COMPLETE.json',dict(status='PASS',created_at_utc=now(),GO=pin(HERE/'GO.json'),workers=commands,elapsed_seconds=time.monotonic()-start,optimizer_updates=1024,training_forwards=1346,development_forwards=800,teacher_forwards=0,network_requests=0,GPU_spend_GBP=0,retries=0))
    except Exception as e:
        save(out/'FAILURE.json',dict(status='STOP',error=type(e).__name__+': '+str(e),commands=commands,elapsed_seconds=time.monotonic()-start));raise
if __name__=='__main__':main()
