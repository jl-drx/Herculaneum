"""Root binds a separate ready preservation plan after technical review."""
import argparse
import json
from pathlib import Path
from control_common import HERE,BASE,shared,record,verify,write_json,require


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--review',type=Path,action='append',required=True); parser.add_argument('--ready',action='store_true')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    basepath=BASE/'final-plan/control-plan.json'; base,_=shared.load_plan(basepath,execute=True)
    historical=dict(frozen_seed42=base['historical_results']['frozen_seed42'],
        **{a:record(BASE/'run'/a/'result.json') for a in ['A_smooth','B_smooth']})
    receipts={a:json.loads(verify(item).read_text()) for a,item in historical.items()}
    logs={a:record(BASE/'run'/a/'steps.jsonl') for a in ['A_smooth','B_smooth']}
    sentinels={a:record(BASE/'run'/a/'first-step-sentinel.json') for a in ['A_smooth','B_smooth']}
    first={a:json.loads(verify(item).read_text().splitlines()[0]) for a,item in logs.items()}
    plan={k:base[k] for k in ['adaptation','development','initializer','teacher','schedule','exposure','helper','metrics','cutoff',
        'calibration','training','limits','diagnostic','diagnostic_role','screens','historical_initial_state_sha256']}
    plan.update(kind='preservation-continuation-v1',ready_for_execution=args.ready,base_plan=record(basepath),
        new_arms=['A_preserve','B_preserve'],historical_results=historical,
        historical_plan_records={a:r['plan'] for a,r in receipts.items()},historical_step_logs=logs,
        historical_first_step=first,historical_first_sentinels=sentinels,
        preservation=dict(coefficient=1.,temperature=1.,dtype='float64',teacher='unchanged seed42 initializer',
            support='A: M; B: M union P; separate masked means',pseudo_coefficient=.25,minimum_mean=-1e-6,
            first_logit_absolute_tolerance=1e-7,first_element_absolute_tolerance=1e-7,
            first_output_gradient_absolute_tolerance=1e-7,first_parameter_norm_absolute_tolerance=1e-6),
        scope='Conventional initialization-prediction preservation on reused development diagnostics; no novelty, new holdout accuracy, significance or reading.',
        logits_contract=dict(shape=[32,2,1,128,128],dtype='float32',timing='Student and frozen teacher before each update; identical input tensor'),
        code={p.name:record(p) for p in sorted(HERE.glob('*.py'))})
    inputs=[*base['verified_inputs'],record(basepath),record(HERE.parent/'PROTOCOL.md'),record(HERE/'code-freeze.json'),
        record(HERE/'ARCHITECTURE_CHECK.md'),record(HERE/'architecture-check.json'),*[record(p) for p in args.review]]
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256','size_bytes'}<=value.keys():
                item={k:value[k] for k in ['path','sha256','size_bytes']}; verify(item); inputs.append(item)
            for child in value.values(): collect(child)
        elif isinstance(value,list):
            for child in value: collect(child)
    collect(plan)
    for receipt in receipts.values(): collect(receipt)
    plan['verified_inputs']=sorted({i['path']:i for i in inputs}.values(),key=lambda i:i['path'])
    write_json(args.output_dir/'control-plan.json',plan); print(json.dumps(record(args.output_dir/'control-plan.json')))


if __name__=='__main__': main()
