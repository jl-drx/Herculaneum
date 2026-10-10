"""Prepare conditional retention using saved JSON/source only; no array/model calls."""
import argparse
import json
from pathlib import Path
from retention_common import HERE, LAB, ARMS, CASE_IDS, LIMITS, ANALYSIS_LIMITS, GATE, SCREEN, ROOT_DESIGN_SHA, verify_gate_result, require, record, identity, read_json, collect, write_json

TENTH_PLAN = dict(path=str(LAB/'reports/tenth-experiment/implementation/final-plan/control-plan.json'),
    size_bytes=109068, sha256='9e2c9f4f778c3f794e103191fd413937b521fa0d0b72cab0e3905778ce3e3a55')
TENTH_METRICS = dict(path=str(LAB/'reports/tenth-experiment/implementation/analysis/metrics.json'),
    size_bytes=57942, sha256='ff8e549c93ae1430373ff956a5448b5e10abf7202ede858828b0f8192362b751')
ELEVENTH_PLAN = dict(path=str(LAB/'reports/eleventh-experiment/comparison/final-plan/comparison-plan.json'),
    size_bytes=28259, sha256='61e82ef922db9e33d5a0c5cc3c69a6bd7d8a0e1bc836d3418f2bd63d8680f45f')
ELEVENTH_METRICS = dict(path=str(LAB/'reports/eleventh-experiment/comparison/analysis/metrics.json'),
    size_bytes=166792, sha256='ea510202c8c95cdea09377213e0103fb6d0a57818b6b57f6699d20289d83d3f7')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gate', type=Path); parser.add_argument('--root-decision', type=Path)
    parser.add_argument('--review', type=Path, action='append', default=[]); parser.add_argument('--ready', action='store_true')
    args = parser.parse_args()
    tenth, tm = read_json(TENTH_PLAN), read_json(TENTH_METRICS)
    eleventh, em = read_json(ELEVENTH_PLAN), read_json(ELEVENTH_METRICS)
    require(tm['status'] == em['status'] == 'PASS' and tm['plan'] == TENTH_PLAN and em['plan'] == ELEVENTH_PLAN,
        'Historical metrics provenance differs')
    require(tenth['historical_initial_state_sha256'] == eleventh['models']['frozen_seed42']['state_sha256'], 'Initializer lineage differs')
    models = {}; eligible = []; gate_record = None; training_plan=None; training_results={}
    if args.gate:
        gate_record = record(args.gate); gate = read_json(gate_record)
        require(gate['status'] == 'PASS' and gate['gate_contract'] == GATE and set(gate['arms']) == set(ARMS), 'Wrong training gate')
        training_plan=identity(gate['plan']); training_contract=read_json(training_plan)
        require(training_contract['kind']=='conservative-fixed-risk-continuation-v1'
            and training_contract['ready_for_execution'] is True and training_contract['base_plan']==TENTH_PLAN
            and training_contract['training']==dict(tenth['training'],learning_rate=1e-6)
            and training_contract['gate']==GATE,'Matched training contract differs')
        for key in ['initializer','helper','schedule','adaptation','preservation','historical_initial_state_sha256']:
            require(training_contract[key]==tenth[key], 'Matched training source differs: '+key)
        for arm in ARMS:
            entry = gate['arms'][arm]; values = entry['fixed_evaluation']
            expected = verify_gate_result(entry,arm,training_plan,tenth['historical_initial_state_sha256'])
            training_results[arm]=identity(entry['training_result'])
            if expected:
                models[arm] = dict(role=arm, learning_rate=1e-6, checkpoint=identity(entry['checkpoint']),
                    state_sha256=entry['final_state_sha256'], training_result=identity(entry['training_result']), fixed_evaluation=values)
                eligible.append(arm)
        require(gate['eligible_arms'] == eligible, 'Gate eligible order differs')
    baseline_lock = eleventh['models']['frozen_seed42']['threshold_lock']
    lock = read_json(baseline_lock); threshold = eleventh['models']['frozen_seed42']['thresholds']['primary']
    require(lock['primary'] == threshold == 0.7558947205543518 and lock['arm'] == 'frozen_seed42', 'Frozen baseline cutoff differs')
    cases = []; historical = {}
    for role in ['frozen_seed42']+ARMS:
        old = read_json(tm['model_results'][role]); require(old['status'] == 'PASS' and old['arm'] == role, 'Old pooled result differs')
        frag = read_json(em['model_results'][role]); require(frag['status'] == 'PASS' and frag['role'] == role
            and frag['state_sha256'] == eleventh['models'][role]['state_sha256'], 'Old Frag4 state differs')
        old_state=old['initial_state_sha256'] if role=='frozen_seed42' else old['training']['final_state_sha256']
        require(old_state==frag['state_sha256'], 'Pooled/Frag4 retained state mismatch')
        historical[role] = dict(pooled_result=identity(tm['model_results'][role]), frag4_result=identity(em['model_results'][role]),
            state_sha256=eleventh['models'][role]['state_sha256'], learning_rate=None if role=='frozen_seed42' else 1e-4)
        for value in tenth['diagnostic']:
            if role == 'frozen_seed42':
                case = dict(id=value['id'], family='pooled21', physical_group='PHerc0814', layer_start=2,
                    legacy_case=value, primary_mask=identity(value['data']['arrays']['primary_annotation_mask']),
                    canonical_counts={k:tm['diagnostic'][value['id']][k] for k in ['fixed_primary_count','positive','negative']},
                    historical_maps={}, fixed_baseline_cutoff=threshold)
                cases.append(case)
            case = next(c for c in cases if c['id'] == value['id'])
            case['historical_maps'][role] = old.get('diagnostic',old.get('fresh',{}))[case['id']]
        for value in eleventh['cases']:
            if role == 'frozen_seed42':
                old_domain = em['per_context']['forward'][value['id']]['domains']['primary']
                cases.append(dict(id=value['id'], family='frag4_float17', physical_group='PHercParis1Fr39', layer_start=0,
                    legacy_case=value, primary_mask=identity(value['arrays']['primary_mask']), historical_maps={},
                    canonical_counts=dict(fixed_primary_count=old_domain['fixed_count'], positive=old_domain['positive'], negative=old_domain['negative']),
                    fixed_baseline_cutoff=threshold))
            next(c for c in cases if c['id']==value['id'])['historical_maps'][role] = frag['runs']['forward'][value['id']]
    cases = [next(c for c in cases if c['id']==name) for name in CASE_IDS]
    require(not args.ready or (args.gate and args.root_decision and args.review), 'Ready requires gate, root decision and review')
    plan = dict(kind='conditional-final-state-retention-v1', ready_for_execution=args.ready, arms=ARMS,
        eligible_arms=eligible, models=models, gate_contract=GATE, screen_contract=SCREEN, training_gate=gate_record,
        training_plan=training_plan,training_results=training_results,
        root_decision=record(args.root_decision) if args.root_decision else None, reviews=[record(p) for p in args.review],
        cases=cases, historical=historical, historical_plans=dict(pooled=TENTH_PLAN,frag4=ELEVENTH_PLAN),
        historical_metrics=dict(pooled=TENTH_METRICS,frag4=ELEVENTH_METRICS),
        baseline_cutoff=dict(value=threshold,lock=identity(baseline_lock),rule='Retained frozen_seed42 conservative-three-development-region primary cutoff; float64(score)>=cutoff; applied equally to all states'),
        initializer=tenth['initializer'], helper=tenth['helper'], shared=eleventh['shared'], metrics=tenth['metrics'], cutoff=tenth['cutoff'],
        legacy_prepare=eleventh['code']['prepare_reference.py'], legacy_worker=eleventh['code']['run_comparison.py'],
        initial_state_sha256=tenth['historical_initial_state_sha256'], limits=LIMITS,analysis_limits=ANALYSIS_LIMITS,
        inference=dict(direction='forward',mask_passed_to_adapter=False,
            normalization='unchanged retained robust_mad 1/99 per 17x128x128 tile',tiles_per_case=25,
            no_baseline_forward=True,no_training=True,no_threshold_fitting=True),
        interpretation='Conditional reused supplied-reference retention diagnostics; known diagnostic/model-selection history, unresolved upstream exposure, one initializer, two contexts per specimen; no fresh holdout, independent accuracy, confidence interval, reading or Plinius method claim.')
    plan['code']={p.name:record(p) for p in sorted(HERE.glob('*.py'))}
    plan['design']=record(HERE/'RETENTION_DESIGN.md')
    plan['research_design']=record(HERE.parent/'RESEARCH_DESIGN.md')
    require(plan['research_design']['sha256']==ROOT_DESIGN_SHA, 'Root scientific design changed')
    plan['code_freeze']=record(HERE/'code-freeze.json') if args.ready else None
    pins={}; collect(plan,pins); plan['verified_inputs']=list(pins.values())
    args.output_dir.mkdir(parents=True,exist_ok=False); output=args.output_dir/'retention-plan.json'; write_json(output,plan)
    print(json.dumps(dict(status='PASS',plan=record(output),ready=args.ready,eligible_arms=eligible,
        actual_array_reads=0,actual_checkpoint_reads=0,model_calls=0)))


if __name__=='__main__': main()
