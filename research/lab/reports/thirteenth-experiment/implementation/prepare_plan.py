"""Root creates a byte-pinned plan; preparation decodes JSON/source only."""
import argparse
import json
from pathlib import Path
from control_common import HERE,TEN,LAB,ARMS,LIMITS,MEASUREMENT,GATE,record,require,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--review',action='append',type=Path,required=True)
    parser.add_argument('--ready',action='store_true'); args=parser.parse_args()
    require(not args.output_dir.exists(),'Fresh plan directory required')
    basepath=TEN/'final-plan/control-plan.json'; base=json.loads(basepath.read_text()); basepin=record(basepath)
    require(base['ready_for_execution'] is True,'Historical plan not ready')
    historical={}; probe={}
    for arm in ARMS:
        path=TEN/'run'/arm/'result.json'; result=json.loads(path.read_text())
        require(result['status']=='PASS' and result['plan']==basepin,'Historical result lineage differs')
        steps=result['training']['steps_log']; rows=[json.loads(x) for x in Path(steps['path']).read_text().splitlines()]
        require(record(steps['path'])==steps and len(rows)==32 and all('input_target_sha256' in r for r in rows),'Historical batch records incomplete')
        historical[arm]=dict(result=record(path),steps=steps,teacher=result['training']['preupdate_logits']['teacher'],
            first_sentinel=result['training']['first_step_sentinel'])
        short=arm.removesuffix('_preserve'); pp=LAB/'reports/twelfth-experiment/step-probe/run'/short/'result.json'
        pr=json.loads(pp.read_text()); trial=pr['trials'][0]
        require(pr['status']=='PASS' and trial['learning_rate']==1e-6,'First-step probe differs')
        probe[arm]=dict(result=record(pp),post_logits=trial['post_logits'],
            post_state_sha256=trial['post_state_sha256'],initial_state_sha256=trial['initial_state_sha256'])
    plan={k:base[k] for k in ['adaptation','initializer','teacher','schedule','exposure','helper','preservation','historical_initial_state_sha256']}
    plan.update(kind='conservative-fixed-risk-continuation-v1',ready_for_execution=args.ready,new_arms=ARMS,
        base_plan=basepin,training=dict(base['training'],learning_rate=1e-6),limits=LIMITS,measurement=MEASUREMENT,gate=GATE,
        historical=historical,first_step_probe=probe,
        shared_sources={name:base['code'][name] for name in ['control_common.py','preservation.py']},
        runtime=dict(python='3.12.14',numpy='2.5.3',torch='2.12.0+cpu'),
        scientific_design=record(HERE.parent/'RESEARCH_DESIGN.md'),reviews=[record(p) for p in args.review],
        code={p.name:record(p) for p in sorted(HERE.glob('*.py'))},
        scope='Fixed adaptation-risk control only; no calibration, retention or heldout inference',
        logits_contract=dict(pre_post_shape=[32,2,1,128,128],fixed_risk_shape=[4,32,2,1,128,128],
            dtype='float32',fixed_risk_update_order=[1,8,16,32]),
        preparation='JSON/source records only; arrays and checkpoints verified by root worker before execution')
    sources=[*base['verified_inputs'],basepin]
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256','size_bytes'}<=value.keys(): sources.append({k:value[k] for k in ['path','sha256','size_bytes']})
            for child in value.values(): collect(child)
        elif isinstance(value,list):
            for child in value: collect(child)
    collect(plan)
    freeze=HERE/'code-freeze.json'
    if args.ready: require(freeze.is_file(),'Code freeze required for a ready plan')
    plan['code_freeze']=record(freeze) if freeze.is_file() else None
    if freeze.is_file(): sources.append(plan['code_freeze'])
    by_path={}
    for item in sources:
        if item['path'] in by_path: require(by_path[item['path']]==item,'Conflicting inherited pin')
        by_path[item['path']]=item
    plan['verified_inputs']=sorted(by_path.values(),key=lambda r:r['path'])
    args.output_dir.mkdir(parents=True); target=args.output_dir/'training-plan.json'
    write_json(target,plan); print(json.dumps(record(target)))


if __name__=='__main__': main()
