"""Publish all saved risk outcomes and the predeclared conditional-retention gate."""
import argparse
import json
from pathlib import Path
from control_common import ARMS,UPDATES,GATE,load_plan,record,verify,write_json,require,average,eligible


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path); parser.add_argument('--output-dir',required=True,type=Path); args=parser.parse_args()
    plan,path=load_plan(args.plan,execute=True); pin=record(path); args.output_dir.mkdir(parents=True,exist_ok=False)
    arms={}; reports={}
    for arm in ARMS:
        target=args.run_dir/arm/'result.json'; receipt=json.loads(target.read_text())
        require(receipt['status']=='PASS' and receipt['arm']==arm and receipt['plan']==pin and receipt['initial_state_sha256']==plan['historical_initial_state_sha256'],'Result lineage differs')
        for item in [receipt['checkpoint'],receipt['steps_log'],receipt['teacher_cache'],receipt['first_step_sentinel'],*receipt['raw_logits'].values()]: verify(item)
        rows=[json.loads(s) for s in verify(receipt['steps_log']).read_text().splitlines()]; require(len(rows)==32,'Steps incomplete')
        initial=receipt['baseline_fixed_evaluation']['equal_batch_means']; require(initial==average(receipt['baseline_fixed_evaluation']['per_batch']),'Baseline reduction differs')
        checkpoints={}
        for update in UPDATES:
            item=receipt['fixed_evaluation'][str(update)]; final=item['equal_batch_means']; require(final==average(item['per_batch']),'Equal-batch reduction differs')
            checkpoints[str(update)]=dict(equal_batch_means=final,deltas_vs_cache_initializer={k:final[k]-initial[k] for k in initial},
                current_batch_post_bitwise_equal=item['current_batch_post_bitwise_equal'])
        final=checkpoints['32']['equal_batch_means']; passed=eligible(initial,final); require(passed==receipt['eligible'],'Worker gate differs')
        arms[arm]=dict(eligible=passed,training_result=record(target),checkpoint=receipt['checkpoint'],final_state_sha256=receipt['final_state_sha256'],
            fixed_evaluation=dict(exposures=32,base_initial=initial['base'],base_final=final['base'],full_initial=initial['full'],full_final=final['full'],
                base_delta=final['base']-initial['base'],full_delta=final['full']-initial['full']))
        reports[arm]=dict(initializer=initial,all_fixed_checkpoints=checkpoints,
            immediate_base_increases=sum(r['post']['base']>r['pre']['base'] for r in rows),
            immediate_full_increases=sum(r['post']['full']>r['pre']['full'] for r in rows),
            online_preupdate_mean={k:sum(r['pre'][k] for r in rows)/32 for k in initial},training_result=record(target))
    gate=dict(status='PASS',plan=pin,gate_contract=GATE,eligible_arms=[a for a in ARMS if arms[a]['eligible']],arms=arms)
    write_json(args.output_dir/'retention-gate.json',gate)
    write_json(args.output_dir/'risk-summary.json',dict(status='PASS',plan=pin,arms=reports,
        scope='Same fixed adaptation batches; no population estimate, heldout risk, calibration or retention score',gate=record(args.output_dir/'retention-gate.json')))
    load_plan(path,execute=True); require(record(path)==pin,'Frozen inputs changed during analysis')
    print(json.dumps(dict(gate=record(args.output_dir/'retention-gate.json'),summary=record(args.output_dir/'risk-summary.json'))))


if __name__=='__main__': main()
