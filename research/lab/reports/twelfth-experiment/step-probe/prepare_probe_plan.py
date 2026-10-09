"""Prepare a text-only matched first-step plan; binary inputs are checked by root's worker."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[2]
BASE = LAB / 'reports/tenth-experiment/implementation/final-plan/control-plan.json'


def record(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(output, ready=False, reviews=(), audits=()):
    output = Path(output)
    if output.exists():
        raise ValueError('Fresh proposal/plan required')
    if ready and (not reviews or not audits):
        raise ValueError('Root-ready plan must bind technical review and offline audit records')
    base = json.loads(BASE.read_text())
    assert base['ready_for_execution'] is True
    by_path = {x['path']: x for x in base['verified_inputs']}
    def inherited(relative):
        path = (LAB / relative).resolve()
        value = by_path[str(path)]
        assert record(path) == value  # Sources only; no binary payload read here.
        return value
    historical = {}
    historical_inputs = []
    for arm in ['A', 'B']:
        path = LAB / f'reports/tenth-experiment/implementation/run/{arm}_preserve/result.json'
        result = json.loads(path.read_text())
        assert result['status'] == 'PASS'
        historical[arm] = {'result': record(path),
                           'initial_logits': result['training']['preupdate_logits'],
                           'first_step_sentinel': result['training']['first_step_sentinel'],
                           'smoothed_first_step': base['historical_first_step'][arm + '_smooth'],
                           'hard_first_step': base['historical_first_sentinels'][arm + '_smooth']}
        historical_inputs += [record(path), result['training']['first_step_sentinel'],
                              base['historical_first_sentinels'][arm + '_smooth']]
        historical_inputs += list(result['training']['preupdate_logits'].values())
    plan = {'schema_version': 1, 'kind': 'matched-single-step-descent-v1',
            'ready_for_execution': ready, 'base_plan': record(BASE),
            'scientific_design': record(LAB / 'reports/twelfth-experiment/DIAGNOSTIC_DESIGN.md'),
            'reviews': [record(p) for p in reviews], 'offline_audits': [record(p) for p in audits],
            'scope': 'Two same-support objective families, three independent first steps each; '
                     'no sequence, checkpoint output, calibration, validation or Frag4 inference',
            'initializer': base['initializer'], 'helper': base['helper'],
            'adaptation': base['adaptation'], 'schedule': base['schedule'],
            'initial_state_sha256': base['historical_initial_state_sha256'],
            'historical': historical,
            'shared': inherited('reports/eighth-experiment/implementation/experiment_common.py'),
            'mechanics': inherited('reports/eighth-experiment/implementation/training_mechanics.py'),
            'smoothing': inherited('reports/ninth-experiment/implementation/smoothing.py'),
            'preservation': base['code']['preservation.py'],
            'code': {name: record(HERE / name) for name in ['prepare_probe_plan.py', 'probe_worker.py']},
            'learning_rates': [1e-6, 1e-5, 1e-4],
            'objectives': {'A': {'manual': 1., 'pseudo': 0.}, 'B': {'manual': 1., 'pseudo': .25}},
            'training': {'optimizer': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8,
                         'weight_decay': 0., 'scheduler': None, 'clip_norm': 1.,
                         'clip_denominator_epsilon': 1e-6, 'seed': 20261008,
                         'smoothing_epsilon': .5, 'steps_per_branch': 1,
                         'fresh_optimizer_per_branch': True, 'shared_fixed_clipped_gradient': True},
            'kl': {'orientation': 'initial_teacher||post_student', 'coefficient': 1.,
                   'temperature': 1., 'dtype': 'float64',
                   'support': 'A:meanM; B:meanM+.25meanP; no unknown pixels'},
            'tolerances': {'historical_loss_absolute': 1e-7, 'historical_gradient_norm_absolute': 1e-6,
                           'historical_logits': 'bitwise identical FP32',
                           'initial_kl_absolute': 1e-12, 'kl_minimum': -1e-6,
                           'adam_update': '4*ULP(initial_fp32)+8*eps_fp32*abs(ideal_delta)+1e-12',
                           'loss_change_identity_absolute': 1e-10},
            'limits': {'worker_wall_seconds': 120, 'total_wall_seconds': 300,
                       'cpu_seconds': 360, 'threads': 4, 'inter_op_threads': 1,
                       'rss_acceptance_bytes': 3 * 1024**3, 'address_space_bytes': 12 * 1024**3,
                       'output_bytes': 64 * 1024**2, 'retries': 0},
            'runtime': {'python': '3.12.14', 'numpy': '2.5.3', 'torch': '2.12.0+cpu'},
            'preparation': 'JSON/source pinning only; no checkpoint, arrays or model execution'}
    inputs = [plan['base_plan'], plan['scientific_design'], plan['initializer'], plan['helper'],
              plan['schedule'], plan['shared'], plan['mechanics'], plan['smoothing'], plan['preservation']]
    inputs += historical_inputs + plan['reviews'] + plan['offline_audits'] + list(plan['code'].values())
    for case in plan['adaptation']:
        inputs += list(case['inputs'].values())
        inputs += [case[k] for k in ['teacher_scores', 'teacher_evaluated', 'teacher_metadata']]
    unique = {x['path']: x for x in inputs}
    plan['verified_inputs'] = list(unique.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    return record(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-plan', required=True)
    parser.add_argument('--ready', action='store_true')
    parser.add_argument('--review', action='append', default=[])
    parser.add_argument('--offline-audit', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_plan, args.ready, args.review, args.offline_audit)))
