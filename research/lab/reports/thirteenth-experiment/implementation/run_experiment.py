"""Root sequentially executes both fixed controls, with no retries."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from control_common import HERE,ARMS,load_plan,record,write_json,require


def offline(*args,**kwargs): raise RuntimeError('Network forbidden in conservative controller')
socket.create_connection=offline; socket.socket.connect=offline; socket.socket.connect_ex=offline


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path); args=parser.parse_args()
    signal.alarm(420); plan,path=load_plan(args.plan,execute=True); pin=record(path); args.output_dir.mkdir(parents=True,exist_ok=False)
    write_json(args.output_dir/'frozen-plan.json',plan); workers=[]; initial=[]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for arm in ARMS:
        started=time.perf_counter()
        with (args.output_dir/(arm+'.log')).open('x') as stream:
            command=[sys.executable,str(HERE/'run_worker.py'),'--plan',str(path),'--arm',arm,'--output-dir',str(args.output_dir/arm)]
            try: exitcode=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=180,check=False).returncode
            except subprocess.TimeoutExpired: exitcode='wall_timeout'
        workers.append(dict(arm=arm,exit_code=exitcode,wall_seconds=time.perf_counter()-started))
        write_json(args.output_dir/'execution.json',dict(plan=pin,workers=workers,retries=0))
        require(exitcode==0,'Worker failed; outputs preserved and no retry: '+arm)
        result=json.loads((args.output_dir/arm/'result.json').read_text()); require(result['status']=='PASS' and result['plan']==pin,'Worker result lineage differs')
        initial.append(result['initial_state_sha256']); load_plan(path,execute=True); require(record(path)==pin,'Plan changed')
        require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=plan['limits']['output_bytes'],'Combined output cap failed')
    require(len(set(initial))==1 and initial[0]==plan['historical_initial_state_sha256'],'Matched initializers differ')
    signal.alarm(0); print(json.dumps(dict(status='PASS',workers=workers,initial_state_sha256=initial[0])))


if __name__=='__main__': main()
