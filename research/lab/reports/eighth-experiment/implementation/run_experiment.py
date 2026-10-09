"""Root-owned sequential execution; no retries, no intermediate selection."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from experiment_common import HERE, ARMS, load_plan, record, write_json, require


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True); args=parser.parse_args()
    plan,path=load_plan(args.plan,execute=True); before=record(path); args.output_dir.mkdir(parents=True,exist_ok=False)
    write_json(args.output_dir/'frozen-plan.json',plan); results=[]
    environment=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for arm in ARMS:
        started=time.perf_counter()
        with (args.output_dir/(arm+'.log')).open('x') as stream:
            command=[sys.executable,str(HERE/'run_worker.py'),'--plan',str(path),'--arm',arm,'--output-dir',str(args.output_dir/arm)]
            try: completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=120,check=False,env=environment); exitcode=completed.returncode
            except subprocess.TimeoutExpired: exitcode='wall_timeout'
        results.append(dict(arm=arm,exit_code=exitcode,wall_seconds=time.perf_counter()-started))
        write_json(args.output_dir/'execution.json',dict(plan=before,workers=results,retries=0))
        require(exitcode==0,'Worker failed; no retry: '+arm)
        size=sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())
        require(size<=plan['limits']['output_bytes'],'Aggregate output cap failed'); load_plan(path,execute=True)
        require(record(path)==before,'Plan changed')
    states=[json.loads((args.output_dir/a/'result.json').read_text())['initial_state_sha256'] for a in ARMS]
    require(len(set(states))==1,'Matched initializer states differ')
    print(json.dumps(dict(status='PASS',workers=results,initial_state_sha256=states[0])))


if __name__=='__main__': main()
