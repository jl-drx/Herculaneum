"""Tiny metadata/risk fixtures only; no Torch import, models or retained arrays."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from control_common import average,eligible,record,write_json,require
spec=importlib.util.spec_from_file_location('conservative_worker_fixture',Path(__file__).with_name('run_worker.py'))
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
tree_hash=worker.tree_hash


class FakeTorch:
    class Tensor:
        pass


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True,type=Path); args=parser.parse_args()
    require(not args.output.exists() and 'torch' not in sys.modules,'Fresh no-Torch fixture required')
    rng=np.random.RandomState(19).get_state(); before=tree_hash(rng,FakeTorch)
    clone=(rng[0],rng[1].copy(),rng[2],rng[3],rng[4]); require(tree_hash(clone,FakeTorch)==before,'Equal synthetic RNG differs')
    changed=list(clone); changed[1]=changed[1].copy(); changed[1][0]^=np.uint32(1)
    require(tree_hash(tuple(changed),FakeTorch)!=before,'Changed RNG not detected')
    require(tree_hash([1,2],FakeTorch)!=tree_hash([12],FakeTorch),'Ambiguous sequence framing')
    require(tree_hash([1,2],FakeTorch)!=tree_hash((1,2),FakeTorch),'Container type lost')
    require(tree_hash({'a':1,'b':2},FakeTorch)==tree_hash({'b':2,'a':1},FakeTorch),'Dictionary order changes state')
    require(tree_hash(np.array([0.],dtype=np.float32),FakeTorch)!=tree_hash(np.array([-0.],dtype=np.float32),FakeTorch),'Signed-zero bits lost')
    rows=[dict(base=1.+i/32,full=2.+i/16) for i in range(32)]
    expected=dict(base=1.+31/64,full=2.+31/32); require(average(rows)==expected,'Equal32-batch reduction differs')
    require(eligible(dict(base=1e-4,full=1e-4),dict(base=0.,full=0.)),'Inclusive exact-delta boundary lost')
    require(not eligible(dict(base=1e-4,full=1e-4),dict(base=0.,full=1e-8)),'Both-risk requirement lost')
    guards=0
    for f in [lambda:average(rows[:-1]),lambda:eligible(dict(base=float('nan'),full=0.),dict(base=0.,full=0.))]:
        try: f()
        except ValueError: guards+=1
        else: raise ValueError('Invalid scalar/domain accepted')
    require(guards==2 and 'torch' not in sys.modules,'Fixture guard failed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,dict(status='PASS',checks=11,invalid_input_guards=guards,
        source=record(__file__),worker=record(Path(__file__).with_name('run_worker.py')),
        common=record(Path(__file__).with_name('control_common.py')),torch_imports=0,source_array_reads=0,model_calls=0))
    print(json.dumps(record(args.output)))


if __name__=='__main__': main()
