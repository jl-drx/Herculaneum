"""Format an already verified phase-16 metrics receipt; stdlib only, no scientific execution."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROLES = ('frozen_seed42', 'A_preserve', 'B_preserve')
MODEL_LABELS = dict(zip(ROLES, ('seed42 baseline', 'A preserve', 'B preserve')))
RULES = ('retained_fourteenth', 'frag3_only')
RULE_LABELS = dict(zip(RULES, ('retained 14', 'Frag3 only')))
EVALUATION = ('frag1-context1', 'frag1-context2', 'frag2-context1', 'frag2-context2')
CALIBRATION = ('frag3-context1', 'frag3-context2')
RATE_KEYS = ('sensitivity', 'specificity', 'precision', 'f1',
             'predicted_positive_fraction', 'balanced_accuracy')
UNDEFINED = 'undefined'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fraction(value, label, nullable=True):
    if value is None:
        require(nullable, label + ' must be defined')
    else:
        require(type(value) in (int, float) and math.isfinite(value)
                and 0 <= value <= 1, 'Invalid fraction: ' + label)
    return value


def pct(value):
    return UNDEFINED if value is None else f'{100 * value:.2f}%'


def decimal(value):
    return UNDEFINED if value is None else f'{value:.4f}'


def yesno(value):
    require(type(value) is bool, 'Expected Boolean')
    return 'yes' if value else 'no'


def csv_value(value):
    if value is None:
        return UNDEFINED
    if type(value) is bool:
        return 'true' if value else 'false'
    return value


def csv_text(rows):
    require(bool(rows), 'Empty CSV')
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: csv_value(value) for key, value in row.items()} for row in rows)
    return output.getvalue()


def table(headers, rows):
    def line(cells):
        return '| ' + ' | '.join(str(cell) for cell in cells) + ' |'
    return '\n'.join([line(headers), line(['---'] * len(headers)),
                      *(line(row) for row in rows)]) + '\n'


def validate(metrics):
    require(metrics['status'] == 'PASS' and tuple(metrics['roles']) == ROLES,
            'Expected complete fixed three-model metrics')
    require(metrics['automatic_promotion'] is False, 'Unexpected promotion flag')
    require(metrics['primary_specificity_transfer_rule'] == 'frag3_only',
            'Primary transfer rule changed')
    require(set(metrics['per_context']) == set(CALIBRATION + EVALUATION),
            'Six-context scope differs')
    for model in ROLES:
        require(set(metrics['thresholds'][model]) == set(RULES), 'Cutoff rules differ')
        for cutoff in metrics['thresholds'][model].values():
            # A conservative nextafter cutoff slightly above 1 is valid.
            require(type(cutoff) in (float, int) and math.isfinite(cutoff), 'Nonfinite cutoff')
    for case_id in EVALUATION:
        case = metrics['per_context'][case_id]
        require(case['role'] == 'evaluation' and case['parent_papyrus'] == 'Paris2',
                'Evaluation parent/role differs')
        expected_fragment = 'PHercParis2Fr47' if case_id.startswith('frag1-') else 'PHercParis2Fr143'
        require(case['fragment'] == expected_fragment, 'Evaluation fragment differs')
        for key in ('pixel_count', 'positive', 'negative'):
            require(type(case[key]) is int and case[key] > 0, 'Invalid fixed support count')
        require(case['positive'] + case['negative'] == case['pixel_count'], 'Support counts disagree')
        for model in ROLES:
            scores = case['models'][model]
            for key in ('auc', 'ap', 'supplied_positive_prevalence'):
                fraction(scores[key], key, nullable=(key != 'supplied_positive_prevalence'))
            for rule in RULES:
                op = scores['operating'][rule]
                for key in RATE_KEYS:
                    fraction(op[key], key)
                counts = op['confusion']
                require(set(counts) == {'tp', 'fp', 'tn', 'fn'} and
                        all(type(n) is int and n >= 0 for n in counts.values()), 'Invalid confusion counts')
                require(counts['tp'] + counts['fn'] == case['positive'] and
                        counts['tn'] + counts['fp'] == case['negative'], 'Confusion support differs')
                require(op['zero_predicted_positives'] is (counts['tp'] + counts['fp'] == 0),
                        'Zero-prediction flag differs')
                exact_pass = 20 * counts['fp'] <= case['negative']
                require(op['specificity_at_least_95'] is exact_pass and
                        metrics['specificity_transfer_checks'][model][rule]['per_context'][case_id] is exact_pass,
                        'Specificity transfer flag differs from counts')
    for model in ROLES:
        summary = metrics['paris2_equal_fragment'][model]
        for key in ('auc', 'ap', 'supplied_positive_prevalence'):
            fraction(summary[key], key)
        for rule in RULES:
            for key in RATE_KEYS:
                fraction(summary['operating'][rule][key], key)
            checks = metrics['specificity_transfer_checks'][model][rule]
            require(set(checks['per_context']) == set(EVALUATION), 'Transfer contexts differ')
            require(checks['all_four'] is all(checks['per_context'].values()), 'Transfer conjunction differs')
        require(metrics['primary_specificity_transfer_all_four'][model] is
                metrics['specificity_transfer_checks'][model]['frag3_only']['all_four'],
                'Primary transfer conjunction differs')


def render(metrics, source):
    validate(metrics)
    threshold_rows = [dict(model=model, retained_fourteenth_cutoff=metrics['thresholds'][model][RULES[0]],
                           frag3_only_cutoff=metrics['thresholds'][model][RULES[1]]) for model in ROLES]
    context_rows = []
    ranking_rows = []
    for case_id in EVALUATION:
        case = metrics['per_context'][case_id]
        for model in ROLES:
            score = case['models'][model]
            ranking_rows.append([case_id, MODEL_LABELS[model], decimal(score['auc']), decimal(score['ap']),
                                 pct(score['supplied_positive_prevalence']), case['positive'], case['negative']])
            for rule in RULES:
                op = score['operating'][rule]
                context_rows.append(dict(context=case_id, fragment=case['fragment'], parent_papyrus='Paris2',
                    model=model, cutoff_rule=rule, cutoff=metrics['thresholds'][model][rule],
                    pixel_count=case['pixel_count'], positive_count=case['positive'], negative_count=case['negative'],
                    roc_auc=score['auc'], average_precision=score['ap'],
                    supplied_positive_prevalence_fraction=score['supplied_positive_prevalence'],
                    **{key + '_count': op['confusion'][key] for key in ('tp', 'fp', 'tn', 'fn')},
                    **{key + '_fraction': op[key] for key in RATE_KEYS},
                    zero_predicted_positives=op['zero_predicted_positives'],
                    specificity_at_least_95=op['specificity_at_least_95']))
    summary_rows = []
    for model in ROLES:
        score = metrics['paris2_equal_fragment'][model]
        for rule in RULES:
            summary_rows.append(dict(scope='Paris2_equal_fragment', model=model, cutoff_rule=rule,
                cutoff=metrics['thresholds'][model][rule], roc_auc=score['auc'], average_precision=score['ap'],
                supplied_positive_prevalence_fraction=score['supplied_positive_prevalence'],
                **{key + '_fraction': score['operating'][rule][key] for key in RATE_KEYS}))
    transfer_rows = []
    for model in ROLES:
        for rule in RULES:
            checks = metrics['specificity_transfer_checks'][model][rule]
            transfer_rows.append(dict(model=model, cutoff_rule=rule,
                **{case: checks['per_context'][case] for case in EVALUATION}, all_four=checks['all_four']))
    blocks = [
        '# Saved calibration-transfer results',
        f"Metrics source: `{source['path']}`; SHA-256 `{source['sha256']}` ({source['size_bytes']} bytes).",
        'This helper only formats saved metrics. Independent numerical verification is a separate prerequisite for interpretation.',
        'AUC and AP use 0–1 units. Markdown rates are percentages rounded to two decimals; CSV rates retain 0–1 fractions. '
        'Counts are exact integers. `undefined` preserves missing metrics and never means zero.',
        '## Locked model cutoffs',
        'The Frag3-only cutoff is the maximum of two negative-only calibration cutoffs from Paris1. '
        'The retained 14 cutoff is each model’s own earlier cutoff. Threshold changes leave its ranking scores unchanged.',
        table(['Model', 'Retained 14 cutoff', 'Frag3-only cutoff'],
              [[MODEL_LABELS[row['model']], repr(row['retained_fourteenth_cutoff']), repr(row['frag3_only_cutoff'])]
               for row in threshold_rows]),
        '## Paris2 context ranking and support',
        table(['Context', 'Model', 'ROC AUC', 'AP', 'Positive prevalence', 'Positive count', 'Negative count'], ranking_rows),
        '## Paris2 context operating points',
        'TP/FP/TN/FN refer to the fixed primary support. “No positives” means zero predicted-positive pixels.',
        table(['Context', 'Model', 'Cutoff rule', 'Sensitivity', 'Specificity', 'TP', 'FP', 'TN', 'FN', 'No positives'],
              [[row['context'], MODEL_LABELS[row['model']], RULE_LABELS[row['cutoff_rule']],
                pct(row['sensitivity_fraction']), pct(row['specificity_fraction']),
                row['tp_count'], row['fp_count'], row['tn_count'], row['fn_count'],
                yesno(row['zero_predicted_positives'])] for row in context_rows]),
        '## Paris2 equal-fragment summary',
        'Each fragment first averages its two context metrics equally; Paris2 then averages the two fragment means equally. '
        'This is a macro average, not pooled pixels. Counts are therefore reported per context above. '
        'A summary metric remains undefined if any contributing metric is undefined.',
        table(['Model', 'Cutoff rule', 'ROC AUC', 'AP', 'Positive prevalence', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Predicted positive', 'Balanced accuracy'],
              [[MODEL_LABELS[row['model']], RULE_LABELS[row['cutoff_rule']], decimal(row['roc_auc']),
                decimal(row['average_precision']), pct(row['supplied_positive_prevalence_fraction']),
                *[pct(row[key + '_fraction']) for key in RATE_KEYS]] for row in summary_rows]),
        '## Specificity transfer checks',
        'Each context passes exactly when `20 × FP ≤ negative count`. All four must pass; '
        'the primary rule is Frag3 only. Passing specificity does not establish useful ink detection; '
        'read sensitivity and zero-positive flags alongside it.',
        table(['Model', 'Cutoff rule', *EVALUATION, 'All four'],
              [[MODEL_LABELS[row['model']], RULE_LABELS[row['cutoff_rule']],
                *['PASS' if row[case] else 'FAIL' for case in EVALUATION],
                'PASS' if row['all_four'] else 'FAIL'] for row in transfer_rows]),
        'Both evaluation fragments come from one parent papyrus, Paris2. Labels were inspected for fixed support gates; '
        'complete upstream model exposure is unknown. These are conditional supplied-reference agreement results, '
        'without independent-scroll replication, significance tests, automatic model promotion, authenticated text or translation.',
    ]
    return {'tables.md': '\n\n'.join(blocks) + '\n', 'cutoffs.csv': csv_text(threshold_rows),
            'contexts.csv': csv_text(context_rows), 'paris2-equal-fragment.csv': csv_text(summary_rows),
            'specificity-transfer.csv': csv_text(transfer_rows)}


def record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--expected-metrics-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = record(args.metrics)
    require(source['sha256'] == args.expected_metrics_sha256, 'Metrics SHA-256 differs')
    metrics = json.loads(Path(source['path']).read_text())
    output = render(metrics, source)
    require(record(args.metrics) == source, 'Metrics changed while formatting')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in output.items():
        (args.output_dir / name).write_text(content)
    receipt = dict(status='PASS', scope='presentation_only', metrics=source,
                   helper=record(__file__), outputs=[record(args.output_dir / name) for name in output],
                   numerical_verification_performed=False, scientific_execution=False,
                   csv_undefined_value=UNDEFINED, csv_rate_units='fraction_0_to_1')
    (args.output_dir / 'presentation-receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status='PASS', receipt=record(args.output_dir / 'presentation-receipt.json'))))


if __name__ == '__main__':
    main()
