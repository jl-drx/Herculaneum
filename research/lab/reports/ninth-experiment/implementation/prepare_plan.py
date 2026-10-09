"""Root builds a separate ready loss-control plan after code/protocol review."""
import argparse
import json
from pathlib import Path
from control_common import HERE,BASE,shared,record,verify,write_json,require


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--review',type=Path,action='append',required=True); parser.add_argument('--ready',action='store_true')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    basepath=BASE/'final-plan/training-plan.json'; base,_=shared.load_plan(basepath,execute=True)
    historical={a:record(BASE/'run'/a/'result.json') for a in ['frozen_seed42','A','B']}
    receipts={a:json.loads(verify(item).read_text()) for a,item in historical.items()}
    require(len({r['initial_state_sha256'] for r in receipts.values()})==1,'Historical initializations differ')
    logs={a:record(BASE/'run'/a/'steps.jsonl') for a in ['A','B']}
    first={a:json.loads(verify(item).read_text().splitlines()[0]) for a,item in logs.items()}
    require(first['A']['patches']==first['B']['patches'] and first['A']['loss_U']==first['B']['loss_U']
        and first['A']['loss_P']==first['B']['loss_P'],'Historical first forwards differ')
    plan={k:base[k] for k in ['adaptation','development','initializer','teacher','schedule','exposure','helper','metrics','cutoff','calibration']}
    plan.update(kind='smoothing-continuation-v1',ready_for_execution=args.ready,base_plan=record(basepath),
        training=dict(base['training'],loss_smoothing=.5),limits=dict(base['limits'],output_bytes=400*1024**2),
        diagnostic=base['fresh'],diagnostic_role='reused_development_retention_diagnostic',new_arms=['A_smooth','B_smooth'],historical_results=historical,
        historical_step_logs=logs,historical_first_step=first,historical_initial_state_sha256=receipts['A']['initial_state_sha256'],
        sentinel_tolerances=dict(historical_absolute=1e-7,analytic_absolute=1e-6),
        screens=dict(specificity_numerator=95,specificity_denominator=100,mitigation_mean_sensitivity_gain_min=.02,
            per_context_sensitivity_delta_min=-.02,per_context_auc_delta_min=-.005,per_context_ap_delta_min=-.01),
        scope='Reused eighth contexts are now development retention diagnostics; BCE target convention only, not the original combined Dice/BCE recipe; no new heldout accuracy, significance or reading claim.',
        formula='epsilon0.5: y_s = 0.25 + 0.5*y only inside unchanged manual/pseudo participation; A_s=U_s, B_s=U_s+0.25*P_s',
        code={p.name:record(p) for p in sorted(HERE.glob('*.py'))})
    inputs=[*base['verified_inputs'],record(basepath),record(HERE.parent/'PROTOCOL.md'),record(HERE/'code-freeze.json'),
        *[record(p) for p in args.review]]
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256','size_bytes'}<=value.keys():
                item={k:value[k] for k in ['path','sha256','size_bytes']}; verify(item); inputs.append(item)
            for child in value.values(): collect(child)
        elif isinstance(value,list):
            for child in value: collect(child)
    collect(plan)
    for a,r in receipts.items(): collect(r)
    plan['verified_inputs']=sorted({i['path']:i for i in inputs}.values(),key=lambda i:i['path'])
    write_json(args.output_dir/'control-plan.json',plan); print(json.dumps(record(args.output_dir/'control-plan.json')))


if __name__=='__main__': main()
