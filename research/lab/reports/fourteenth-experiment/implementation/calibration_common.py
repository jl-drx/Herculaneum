"""Pinned narrow contracts and exact descriptive calibration screens."""
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
LAB=Path('/workspace/herculaneum-lab')
PREVIOUS=LAB/'reports/thirteenth-experiment/retention/retention_common.py'
if hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()!='52de8e0fc69f0f45048d726776ee14525d2a4cd4bc322d7576fb129e2338ae60':
    raise ValueError('Frozen retention helper changed')
spec=importlib.util.spec_from_file_location('fourteenth_retention_helpers',PREVIOUS)
previous=importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
record,identity,verify,read_json,write_json,module,collect,require=(getattr(previous,k) for k in
    ['record','identity','verify','read_json','write_json','module','collect','require'])
ARMS=['A_preserve','B_preserve']; ROLES=['frozen_seed42']+ARMS
DEVELOPMENT_IDS=['pherc1667-w029','pherc0814-46527-rank01','pherc0814-46527-rank07']
NEGATIVE_COUNTS=dict(zip(DEVELOPMENT_IDS,[9378,40400,14644]))
DESIGN_SHA='96ed37ff57dbcff80bb9cbd6e984a2bb27ee0f6bf2d7de079dac42adf0b0dc45'
SELECTOR_SHA='38a0ebbab0835845ee2213c2be9505cf9b41a129ab13dd6668ff797bfacacb73'
BASELINE_CUTOFF=0.7558947205543518
LIMITS=dict(worker_wall_seconds=90,worker_cpu_seconds=240,total_wall_seconds=240,threads=4,
    rss_bytes=3*1024**3,address_space_bytes=12*1024**3,output_bytes=32*1024**2,maximum_tile_forwards=118,retries=0)
ANALYSIS_LIMITS=dict(wall_seconds=120,cpu_seconds=120,rss_bytes=1024**3,address_space_bytes=4*1024**3,output_bytes=16*1024**2)
THRESHOLD_CONTRACT=dict(primary='max of all three development-region thresholds',secondary='pherc1667-w029 only',
    specificity_numerator=95,specificity_denominator=100,threshold_operator='float64(score) >= threshold',
    selector='unchanged seventh operating_point.py',diagnostic_fitting=False)


def fraction_record(value):
    return dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def sensitivity(confusion):
    return Fraction(confusion['tp'],confusion['tp']+confusion['fn'])


def recovery_screen(candidate,baseline):
    """Two equal-context rows; all primary conditions remain separate and exact."""
    require(set(candidate)==set(baseline) and len(candidate)==2,'Each specimen needs its two fixed contexts')
    specificity={}; context={}; candidate_mean=Fraction(0); baseline_mean=Fraction(0)
    for name in candidate:
        cc,bc=candidate[name],baseline[name]
        require(all(type(v) is int and v>=0 for row in [cc,bc] for v in row.values()),'Noninteger confusion')
        require(cc['tp']+cc['fn']==bc['tp']+bc['fn']>0 and cc['fp']+cc['tn']==bc['fp']+bc['tn']>0,'Confusion support differs')
        negative=cc['fp']+cc['tn']; delta=sensitivity(cc)-sensitivity(bc)
        specificity[name]=dict(false_positives=cc['fp'],negative_count=negative,passed=20*cc['fp']<=negative)
        context[name]=dict(delta=fraction_record(delta),minimum=fraction_record(Fraction(-1,50)),passed=delta>=Fraction(-1,50))
        candidate_mean+=sensitivity(cc)/2; baseline_mean+=sensitivity(bc)/2
    mean_check=dict(candidate=fraction_record(candidate_mean),baseline=fraction_record(baseline_mean),
        delta=fraction_record(candidate_mean-baseline_mean),passed=candidate_mean>=baseline_mean)
    return dict(specificity_by_context=specificity,mean_sensitivity_comparison=mean_check,
        per_context_sensitivity_checks=context,passed=all(v['passed'] for v in specificity.values())
        and mean_check['passed'] and all(v['passed'] for v in context.values()),automatic_promotion=False)


def load_plan(path,execute=True):
    path=Path(path).resolve(strict=True); plan=json.loads(path.read_text())
    require(plan['kind']=='development-only-calibration-control-v1','Wrong calibration plan')
    if execute: require(plan['ready_for_execution'] is True and plan['reviews'],'Calibration not reviewed ready')
    require(plan['arms']==ARMS and plan['roles']==ROLES and plan['limits']==LIMITS
        and plan['analysis_limits']==ANALYSIS_LIMITS and plan['threshold_contract']==THRESHOLD_CONTRACT,'Frozen scope differs')
    require(plan['research_design']['sha256']==DESIGN_SHA and plan['cutoff']['sha256']==SELECTOR_SHA,'Scientific design or selector changed')
    eighth=read_json(plan['eighth_plan']); retention=read_json(plan['retention_plan'])
    require(plan['development']==eighth['development'] and [c['id'] for c in plan['development']]==DEVELOPMENT_IDS,'Literal development records changed')
    require(plan['diagnostic_cases']==retention['cases'] and plan['models']==retention['models']
        and retention['eligible_arms']==ARMS,'Final states/canonical diagnostic cases changed')
    baseline=read_json(plan['baseline_result'])
    require(baseline['status']=='PASS' and baseline['arm']=='frozen_seed42' and baseline['plan']==plan['eighth_plan']
        and plan['initial_state_sha256']==baseline['initial_state_sha256']==retention['initial_state_sha256'],'Initializer lineage differs')
    require(plan['initializer']==eighth['initializer']==retention['initializer'],'Initializer checkpoint lineage differs')
    require(plan['baseline_cutoff']==BASELINE_CUTOFF and read_json(plan['baseline_lock'])['primary']==BASELINE_CUTOFF,'Baseline cutoff changed')
    require(plan['inference']==dict(layer_start=2,direction='forward',mask_passed_to_adapter=False,
        patch_shape=[17,128,128],stride=64,tiles_by_case=[9,25,25],forwards_per_arm=59,
        normalization='unchanged robust_mad 1/99',baseline_forwards=0,diagnostic_forwards=0,backwards=0),'Inference contract differs')
    for item in plan['verified_inputs']: verify(item)
    for name,item in plan['code'].items(): require(verify(item)==HERE/name,'Calibration execution source differs')
    if execute:
        freeze=read_json(plan['code_freeze']); require(freeze['no_further_edits'] is True,'Source package unsealed')
        for item in freeze['files']:
            verify(item)
            if Path(item['path']).suffix=='.py': require(plan['code'][Path(item['path']).name]==item,'Source seal differs')
    return plan,path


def development_data(plan,case):
    import numpy as np
    shared=module(plan['shared'],'fourteenth_development_contract'); data,labels,primary=shared.evaluation(case)
    shape=(256,256) if case['id']=='pherc1667-w029' else (384,384)
    require(case['evaluation_participation_role']=='public_validation' and data['surface'].shape==(21,*shape)
        and data['surface'].dtype==np.uint8 and labels.dtype==np.bool_ and primary.shape==shape,'Calibration input contract differs')
    require(int((primary&~labels).sum())==NEGATIVE_COUNTS[case['id']],'Canonical negative count changed; no shrinking')
    return data['surface'],labels,primary


def selection_from_map(plan,case,item,selector):
    import numpy as np
    _,labels,primary=development_data(plan,case)
    scores=np.load(verify(item['scores']),allow_pickle=False); evaluated=np.load(verify(item['evaluated']),allow_pickle=False)
    require(scores.dtype==np.float32 and scores.shape==labels.shape and evaluated.dtype==np.bool_ and evaluated.shape==labels.shape,
        'Development score/coverage contract differs')
    require(evaluated[primary].all() and np.isfinite(scores[primary]).all(),'Fixed calibration support unscored/nonfinite; no shrinking')
    selection=selector.choose_development_threshold(scores[primary&~labels])
    require(selection['negative_count']==NEGATIVE_COUNTS[case['id']],'Selector negative count changed')
    return selection
