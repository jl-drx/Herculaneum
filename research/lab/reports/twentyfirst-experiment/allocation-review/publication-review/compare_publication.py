"""Compare the root's published metrics with the independent saved-output report."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve(strict=True)
    require(path.suffix == '.json', 'This comparison reads only JSON reports')
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def close(actual, expected, label, tolerance=1e-12):
    if expected is None:
        require(actual is None, 'Undefined metric differs: ' + label)
    else:
        require(actual is not None and math.isfinite(actual) and abs(actual - expected) <= tolerance,
                'Published metric differs: ' + label)


def mean(values):
    return None if any(value is None for value in values) else math.fsum(values) / len(values)


def from_independent(model):
    rank, operating = model['rank'], model['operating']
    return dict(auc=rank['auc'], ap=rank['ap'], sensitivity=operating['recall'], specificity=operating['specificity'],
                precision=operating['precision'], balanced_accuracy=mean([operating['recall'], operating['specificity']]),
                f1=operating['f1'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('published', 'independent', 'training'):
        parser.add_argument('--' + name, required=True, type=Path)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pins = {name: record(getattr(args, name)) for name in ('published', 'independent', 'training')}
    for name, pin in pins.items():
        require(pin['sha256'] == getattr(args, name + '_sha256'), 'Entry report pin differs: ' + name)
    published, independent, training = [json.loads(Path(pins[name]['path']).read_text())
                                       for name in ('published', 'independent', 'training')]
    require(published['status'] == independent['status'] == training['status'] == 'PASS' and
            published['plan'] == independent['plan'] == training['plan'] and
            published['training_results'] == training['training_results'] and
            published['development_results'] == independent['development_results'] and
            independent['training_verification'] == pins['training'], 'Publication/review lineage differs')
    require(published['threshold'] == .7558947205543518 and published['automatic_promotion'] is False and
            published['new_model_forwards'] == published['threshold_fits'] == published['network_calls'] == 0,
            'Publication scope differs')
    role_map = {'frozen_seed42': 'unchanged_seed42', 'manual_double': 'manual_double', 'manual_retired': 'manual_retired'}
    cases = {row['case']: row for row in independent['cases']}
    require(set(published['per_case']) == set(cases), 'Published case set differs')
    expected_cases = {}
    for name, row in published['per_case'].items():
        truth = cases[name]
        require(row['fragment'] == truth['fragment'] and row['parent_papyrus'] == truth['parent'] and
                row['role'] == 'reused development diagnostic' and set(row['models']) == set(role_map),
                'Published case metadata differs: ' + name)
        expected_cases[name] = {}
        for role, source_role in role_map.items():
            expected = from_independent(truth['models'][source_role])
            expected_cases[name][role] = expected
            actual = row['models'][role]
            for key, value in expected.items():
                close(actual[key], value, name + '/' + role + '/' + key)
            op = truth['models'][source_role]['operating']
            require(actual['confusion'] == {key: op[key] for key in ('tp', 'fp', 'tn', 'fn')},
                    'Published confusion counts differ')
            rank = truth['models'][source_role]['rank']
            require(row['positive'] == rank['positive'] and row['negative'] == rank['negative'],
                    'Published participant counts differ')
    expected_fragments = {}
    for fragment in {row['fragment'] for row in cases.values()}:
        selected = [name for name, row in cases.items() if row['fragment'] == fragment]
        require(len(selected) == 2, 'Fragment must contain both fixed contexts')
        expected_fragments[fragment] = {role: {key: mean([expected_cases[name][role][key] for name in selected])
                                               for key in expected_cases[selected[0]][role]}
                                         for role in role_map}
    require(set(published['per_fragment']) == set(expected_fragments), 'Published fragment set differs')
    for fragment, roles in expected_fragments.items():
        for role, metrics in roles.items():
            for key, value in metrics.items():
                close(published['per_fragment'][fragment][role][key], value, fragment + '/' + role + '/' + key)
    for role in role_map:
        for key in expected_fragments['PHercParis1Fr34'][role]:
            expected = mean([expected_fragments[fragment][role][key] for fragment in ('PHercParis2Fr47', 'PHercParis2Fr143')])
            close(published['Paris2_equal_fragment'][role][key], expected, 'Paris2/' + role + '/' + key)
            close(published['Paris1_equal_context'][role][key], expected_fragments['PHercParis1Fr34'][role][key],
                  'Paris1/' + role + '/' + key)
    for arm in ('manual_double', 'manual_retired'):
        actual, truth = published['training'][arm], training['arms'][arm]
        require(actual['initial'] == truth['producer_baseline'] and
                actual['final'] == truth['all_fixed_checkpoints']['32']['producer_equal_batch_means'],
                'Published training risk differs from verified producer scalars')
        for update, row in truth['all_fixed_checkpoints'].items():
            require(actual['fixed_updates'][update] == row['producer_equal_batch_means'], 'Published fixed update differs')
        for key in ('base', 'full'):
            require(actual['improvement'][key] == actual['initial'][key] - actual['final'][key],
                    'Published training improvement arithmetic differs')
    own=independent['declared_scalar_screen']
    expected_mechanism={arm+'_own_full_improvement':published['training'][arm]['initial']['full']-published['training'][arm]['final']['full'] for arm in ('manual_double','manual_retired')}
    control,candidate=published['training']['manual_double'],published['training']['manual_retired']
    expected_mechanism.update(candidate_manual_BCE_improvement=candidate['initial']['base']-candidate['final']['base'],
        candidate_retired_KL_reduction_vs_control=control['final']['kl_retired']-candidate['final']['kl_retired'],
        candidate_manual_BCE_excess_margin=control['final']['base']+.001-candidate['final']['base'])
    expected_practical={}
    for metric in ('auc','ap'):
        expected_practical['Paris2_candidate_'+metric+'_gain_vs_seed42']=published['Paris2_equal_fragment']['manual_retired'][metric]-published['Paris2_equal_fragment']['frozen_seed42'][metric]
    for name,row in published['per_case'].items():
        for metric in ('auc','ap'):
            expected_practical[name+'_'+metric+'_retention']=row['models']['manual_retired'][metric]-row['models']['frozen_seed42'][metric]
    ambiguities=[]
    for group,expected_values in [('mechanism',expected_mechanism),('practical',expected_practical)]:
        require(set(published[group+'_screen'])==set(expected_values)==set(own[group]['components']), 'Published allocation component set differs')
        for name,row in published[group+'_screen'].items():
            expected=own[group]['components'][name]
            require(row['required_minimum']==expected['required_minimum'] and row['value']==expected_values[name] and
                    row['margin']==row['value']-row['required_minimum'] and row['passed']==(row['value']>=row['required_minimum']),
                    'Published gate arithmetic differs: '+group+'/'+name)
            close(row['value'],expected['value'],group+'/'+name)
            if row['passed']!=expected['passed']:ambiguities.append(group+'/'+name)
        require(published[group+'_screen_all']==all(row['passed'] for row in published[group+'_screen'].values()),
                'Published joint allocation screen differs: '+group)
    plan_pin=record(published['plan']['path']);require(plan_pin==published['plan'],'Plan changed')
    plan=json.loads(Path(plan_pin['path']).read_text())
    historical_pin=record(plan['historical_phase20_metrics']['path'])
    require(historical_pin==plan['historical_phase20_metrics'] and historical_pin['sha256']=='60bb52efcff810830543941e5081f61e9570cb598a772b97c4c2633a1cce0be6',
            'Historical descriptive reference anchor differs')
    historical=json.loads(Path(historical_pin['path']).read_text())
    retained=published['historical_phase20_explicit_depth']
    require(retained['source']==historical_pin and retained['training']==historical['training']['explicit_depth'] and
            retained['Paris2']==historical['Paris2_equal_fragment']['explicit_depth'] and
            retained['Paris1']==historical['Paris1_equal_context']['explicit_depth'] and
            retained['scope']=='Historical descriptive reference only; different own objective B_M+K_M; no new predictions or reoptimization',
            'Historical descriptive reference changed or conflated with current objectives')
    pins.update(plan=plan_pin,historical_phase20_metrics=historical_pin)
    for pin in pins.values():
        require(record(pin['path']) == pin, 'Report changed during comparison')
    report = dict(status='PASS', inputs=pins, per_context_rank_and_operating_models=18,
                  equal_context_and_equal_fragment_macro_verified=True, mechanism_components=5, practical_components=14,
                  published_screen_arithmetic_exact=True, mechanism_screen_all=published['mechanism_screen_all'],practical_screen_all=published['practical_screen_all'],historical_reference_verified=True,
                  producer_independent_gate_ambiguities=ambiguities,
                  scientific_array_reads=0, model_calls=0, checkpoint_decodes=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = args.output_dir/'comparison.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS', comparison=record(output)), sort_keys=True))


if __name__ == '__main__':
    main()
