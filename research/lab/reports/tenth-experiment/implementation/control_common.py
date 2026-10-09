"""Tenth preservation control bound to immutable ninth contracts and helpers."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'ninth-experiment/implementation'
for name,digest in {'control_common.py':'417b80fa3aaf1af6906c03730806756455a851fc00d6bbb3f2c433c6f206dfdd',
    'smoothing.py':'bfb66a0323a50850ebaa13c93ef4d02b6ee70289aa35bae57fa1d87b49f1ea56'}.items():
    if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=digest: raise ValueError('Immutable ninth helper changed')
spec=importlib.util.spec_from_file_location('ninth_shared',BASE/'control_common.py')
shared=importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
sys.path.insert(0,str(BASE))
record,verify,write_json,module,require,training_region,evaluation,state_hash=(getattr(shared,name) for name in
    ['record','verify','write_json','module','require','training_region','evaluation','state_hash'])
ARMS=['frozen_seed42','A_smooth','B_smooth','A_preserve','B_preserve']


def load_plan(path,execute=False):
    path=Path(path).resolve(strict=True); plan=json.loads(path.read_text())
    require(plan['kind']=='preservation-continuation-v1','Wrong preservation control')
    if execute: require(plan['ready_for_execution'] is True,'Preservation not ready')
    for item in plan['verified_inputs']: verify(item)
    for name,item in plan['code'].items(): require(verify(item)==HERE/name,'Execution source differs')
    base,_=shared.load_plan(verify(plan['base_plan']),execute=True)
    for key in ['adaptation','development','initializer','teacher','schedule','exposure','helper','metrics','cutoff','calibration',
        'training','limits','diagnostic','diagnostic_role','screens','historical_initial_state_sha256']:
        require(plan[key]==base[key],'Matched ninth contract changed: '+key)
    require(plan['new_arms']==['A_preserve','B_preserve'],'New arm scope changed')
    require(plan['preservation']==dict(coefficient=1.,temperature=1.,dtype='float64',teacher='unchanged seed42 initializer',
        support='A: M; B: M union P; separate masked means',pseudo_coefficient=.25,minimum_mean=-1e-6,
        first_logit_absolute_tolerance=1e-7,first_element_absolute_tolerance=1e-7,
        first_output_gradient_absolute_tolerance=1e-7,first_parameter_norm_absolute_tolerance=1e-6),'Preservation contract changed')
    for arm,item in plan['historical_results'].items():
        receipt=json.loads(verify(item).read_text()); require(receipt['status']=='PASS' and receipt['arm']==arm
            and receipt['plan']==plan['historical_plan_records'][arm]
            and receipt['initial_state_sha256']==plan['historical_initial_state_sha256'],'Historical arm mismatch')
    return plan,path
