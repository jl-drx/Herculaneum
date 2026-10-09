"""Metadata-only phase18 plan writer; scientific arrays/checkpoints are not read."""
import argparse
import json
from pathlib import Path

from depth_common import HERE, RUNTIME, require, record, read_json, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path,help='Root frozen architecture/training/limits/gates JSON')
    parser.add_argument('--code-freeze',required=True,type=Path)
    parser.add_argument('--source-go',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    reports=HERE.parents[1]
    training_pin=record(reports/'thirteenth-experiment/implementation/final-plan/training-plan.json')
    inference_pin=record(reports/'sixteenth-experiment/inference/final-plan/comparison-plan.json')
    base=read_json(training_pin);inference=read_json(inference_pin)
    config_pin=record(args.config);config=read_json(config_pin)
    require(set(config)=={'architecture','training','limits','gates'},'Root config sections differ')
    source_paths={
        'experiment_common.py':reports/'eighth-experiment/implementation/experiment_common.py',
        'training_mechanics.py':reports/'eighth-experiment/implementation/training_mechanics.py',
        'run_ink_reference.py':HERE.parents[2]/'run_ink_reference.py',
        'normalization.py':HERE.parents[2]/'vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py',
    }
    cases=[];receipts={}
    for phase in ['calibration','evaluation']:
        pin=record(reports/f'sixteenth-experiment/inference/run/{phase}/frozen_seed42/result.json')
        receipt=read_json(pin);receipts[phase]=pin
        require(receipt['status']=='PASS' and receipt['state_sha256']==base['historical_initial_state_sha256'],
                'Historical frozen backbone result differs')
        for case in inference['cases']:
            if case['id'] not in receipt['runs']:continue
            metadata_path=Path(receipt['runs'][case['id']]['logits']['path']).parent/'metadata.json'
            metadata_pin=record(metadata_path);metadata=read_json(metadata_pin)
            require(metadata==receipt['runs'][case['id']], 'Cached inference receipt differs')
            cases.append(dict(case, baseline_metadata=metadata_pin, baseline_result=pin,
                              phase18_role='reused_development_diagnostic'))
    plan=dict(kind='phase18-depth-residual-v1', ready_for_execution=True, runtime=RUNTIME,
              protocol=record(args.protocol), configuration=config_pin,
              code_freeze=record(args.code_freeze), source_go_path=str(args.source_go.resolve()),
              code={p.name:record(p) for p in sorted(HERE.glob('*.py'))},
              source_code={name:record(path) for name,path in source_paths.items()},
              arms=['raw_control','explicit_depth'], training_base_plan=training_pin,
              inference_base_plan=inference_pin, schedule=base['schedule'],
              backbone_state_sha256=base['historical_initial_state_sha256'],
              backbone_checkpoint_lineage=base['initializer'],
              teacher_caches={a:base['historical'][a]['teacher'] for a in ['A_preserve','B_preserve']},
              historical_step_logs={a:base['historical'][a]['steps'] for a in ['A_preserve','B_preserve']},
              historical_training_results={a:base['historical'][a]['result'] for a in ['A_preserve','B_preserve']},
              historical_inference_results=receipts,cases=cases,**config)
    require([c['id'] for c in cases]==['frag3-context1','frag3-context2','frag1-context1','frag1-context2','frag2-context1','frag2-context2'],
            'Fixed six-context order differs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,plan)
    print(json.dumps(dict(status='PASS',plan=record(args.output),scientific_array_reads=0,checkpoint_reads=0)))


if __name__=='__main__':main()
