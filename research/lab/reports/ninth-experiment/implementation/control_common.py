"""Narrow ninth control, bound to immutable eighth inputs and mechanics."""
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'eighth-experiment/implementation'
for name,digest in {'experiment_common.py':'8d933dae02248d7b5fbbd26370c95fca53191bcff4e1dc51e984c40d8095b8cf',
    'training_mechanics.py':'83dc4482381d4efeb79382e4698844116c90b02f3370e15940a912021ddf7e1b'}.items():
    if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=digest: raise ValueError('Immutable eighth helper changed')
sys.path.insert(0,str(BASE))
import experiment_common as shared
from experiment_common import record,verify,write_json,module,require,training_region,evaluation,state_hash
ARMS=['frozen_seed42','A','B','A_smooth','B_smooth']


def load_plan(path,execute=False):
    path=Path(path).resolve(strict=True); plan=json.loads(path.read_text())
    require(plan['kind']=='smoothing-continuation-v1','Wrong loss control')
    if execute: require(plan['ready_for_execution'] is True,'Control not ready')
    for item in plan['verified_inputs']: verify(item)
    for name,item in plan['code'].items(): require(verify(item)==HERE/name,'Control source differs')
    base,_=shared.load_plan(verify(plan['base_plan']),execute=True)
    for key in ['adaptation','development','initializer','teacher','schedule','exposure','helper','metrics','cutoff','calibration']:
        require(plan[key]==base[key],'Matched eighth contract changed: '+key)
    expected=dict(base['training'],loss_smoothing=.5)
    require(plan['training']==expected and plan['limits']==dict(base['limits'],output_bytes=400*1024**2),'Control settings changed')
    require(plan['diagnostic']==base['fresh'] and plan['new_arms']==['A_smooth','B_smooth'],'Diagnostic/arm scope changed')
    require(plan['diagnostic_role']=='reused_development_retention_diagnostic','Diagnostic role changed')
    require(plan['sentinel_tolerances']==dict(historical_absolute=1e-7,analytic_absolute=1e-6),'Sentinel tolerance changed')
    require(plan['screens']==dict(specificity_numerator=95,specificity_denominator=100,mitigation_mean_sensitivity_gain_min=.02,
        per_context_sensitivity_delta_min=-.02,per_context_auc_delta_min=-.005,per_context_ap_delta_min=-.01),'Descriptive screens changed')
    require(plan['schedule']['sha256']=='87bb53a0a80ba2268b1b2b020342058c2facd44f343c53b8a5ac478da06e40e1','Schedule changed')
    for arm,item in plan['historical_results'].items():
        receipt=json.loads(verify(item).read_text()); require(receipt['status']=='PASS' and receipt['arm']==arm
            and receipt['plan']==plan['base_plan'] and receipt['initial_state_sha256']==plan['historical_initial_state_sha256'],'Historical arm mismatch')
    return plan,path
