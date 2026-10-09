#!/usr/bin/env python3
"""Common-domain descriptive analysis for the frozen four-arm w029 comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from binary_metrics import binary_metrics, paired_score_diagnostics
from comparison_common import (ARMS, class_counts, import_helper, load_data, load_plan,
                               record, require, sufficient, verify_record, write_json)


def plot_comparison(directory, arrays, labels, primary, secondary, scores, curves, data_contract):
    import os
    os.environ['MPLCONFIGDIR'] = str(directory/'.matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].imshow(arrays['surface'][10], cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Pooled center layer 10; source intensity')
    annotation = np.zeros((*labels.shape, 3), dtype=np.float32)
    annotation[secondary & ~labels] = [.25, .25, .7]
    annotation[secondary & labels] = [1, .8, 0]
    axes[0, 1].imshow(annotation)
    axes[0, 1].contour(primary, levels=[.5], colors=['white'], linewidths=.5)
    axes[0, 1].set_title('Supported supplied labels; white primary edge')
    axes[0, 1].legend(handles=[Patch(color=[1, .8, 0], label='Supplied positive'),
        Patch(color=[.25, .25, .7], label='Supplied negative'),
        Patch(color='black', label='Excluded / unsupported validation'),
        Line2D([0], [0], color='white', label='Primary domain edge')],
        loc='lower left', fontsize=6, facecolor='#555555', labelcolor='white', framealpha=.9)
    for axis, label in zip([axes[0, 2], *axes[1]], ARMS):
        image = axis.imshow(scores[label], cmap='viridis', vmin=0, vmax=1)
        axis.contour(primary, levels=[.5], colors=['white'], linewidths=.5)
        axis.set_title(label + '\nRaw score; fixed 0–1 scale')
    for axis in axes.flat:
        axis.set_axis_off()
    fig.colorbar(image, ax=list(axes.flat), shrink=.65, label='Uncalibrated model score')
    bbox = data_contract['bbox_source_xywh']
    spacing = data_contract['pixel_spacing_um']
    scale = f'Verified XY spacing {spacing} µm/pixel' if spacing is not None else 'Physical XY scale unverified'
    fig.suptitle('Supplied-label checkpoint diagnostic; no authenticated geometry or reading\n'
        f"{data_contract['segment']}; public source crop XYWH {bbox}; {scale}", fontsize=10)
    fig.savefig(directory/'comparison.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label in ARMS:
        curve = curves[label]
        axes[0].plot(curve['fpr'], curve['tpr'], label=label)
        axes[1].plot(curve['recall'], curve['precision'], label=label)
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=.7)
    axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title='Fixed primary ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Fixed primary precision–recall')
    for axis in axes:
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=7)
    fig.savefig(directory/'primary-curves.png', dpi=140)
    plt.close(fig)


def analyze(plan_path, output):
    import json
    plan, plan_path = load_plan(plan_path, execute=True)
    output = Path(output).resolve(strict=True)
    require((output/'frozen-plan.json').read_bytes() == plan_path.read_bytes(), 'Executing plan changed')
    arrays, labels, primary, secondary, preflight = load_data(plan)
    directory = output/'analysis'
    directory.mkdir(exist_ok=False)
    scores, metadata, evaluated = {}, {}, {}
    common = np.ones(labels.shape, dtype=bool)
    for label in ARMS:
        item = json.loads((output/label/'metadata.json').read_text())
        require(item['label'] == label and item['plan']['sha256'] == record(plan_path)['sha256'], 'Model output plan mismatch')
        require(item['inputs'] == plan['inputs'] and item['execution_code'] == plan['code'], 'Model source/code contract mismatch')
        require(item['model']['strict_state_match'] is True
                and not item['model']['missing_keys'] and not item['model']['unexpected_keys'], 'Missing strict load evidence')
        scores[label] = np.load(verify_record(item['raw_scores']), allow_pickle=False)
        evaluated[label] = np.load(verify_record(item['evaluated_mask']), allow_pickle=False)
        require(scores[label].dtype == np.float32 and scores[label].shape == labels.shape
                and evaluated[label].dtype == np.bool_ and evaluated[label].shape == labels.shape,
                'Model output array contract mismatch')
        finite = np.isfinite(scores[label])
        require(np.all((scores[label][finite] >= 0) & (scores[label][finite] <= 1)), 'Scores outside [0,1]')
        common &= finite & evaluated[label]
        metadata[label] = item
    helper = import_helper(plan)
    pair_audit = helper.audit_matched_pair(metadata['russo_control75k']['model']['embedded_config'],
                                         metadata['russo_ex01675k']['model']['embedded_config'])
    domains = dict(primary=primary & common, secondary=secondary & common)
    for name, mask in domains.items():
        np.save(directory/f'{name}-common-mask.npy', mask, allow_pickle=False)
    counts = {name: class_counts(labels, mask) for name, mask in domains.items()}
    support = dict(preflight=preflight, final_domains=counts,
        primary_pixels_lost_to_common_model_support=int((primary & ~common).sum()),
        secondary_pixels_lost_to_common_model_support=int((secondary & ~common).sum()),
        matched_pair_config_audit=pair_audit,
        evaluated_pixels_by_model={label: int(evaluated[label].sum()) for label in ARMS},
        observed_occupancy='np.any(surface[2:19] != 0, axis=0), diagnostic only; not excluded from metrics.',
        mask_records={name: record(directory/f'{name}-common-mask.npy') for name in domains})
    write_json(directory/'common-support.json', support)
    require(sufficient(counts['primary']), 'STOP: common model support lacks >=256 supplied pixels of each class; no metric-domain relaxation')
    metrics, curve_records, primary_curves = {}, {}, {}
    occupancy = arrays['source_observed_nonzero_mask']
    for name, mask in domains.items():
        result, curve_records[name] = {}, {}
        for label in ARMS:
            result[label], curve = binary_metrics(labels[mask], scores[label][mask])
            curve_path = directory/f'{name}-{label}-curves.npz'
            np.savez_compressed(curve_path, **curve)
            curve_records[name][label] = record(curve_path)
            if name == 'primary':
                primary_curves[label] = curve
            result[label]['occupancy_strata'] = {}
            for stratum, stratum_mask in [('nonzero_profiles', mask & occupancy), ('observed_zero_profiles', mask & ~occupancy)]:
                result[label]['occupancy_strata'][stratum], _ = binary_metrics(labels[stratum_mask], scores[label][stratum_mask])
        paired = paired_score_diagnostics(scores['russo_ex01675k'][mask], scores['russo_control75k'][mask], labels[mask])
        for metric in ['roc_auc', 'average_precision', 'balanced_accuracy']:
            first, second = result['russo_ex01675k'][metric], result['russo_control75k'][metric]
            paired['ex016_minus_control_' + metric] = first - second if first is not None and second is not None else None
        metrics[name] = dict(models=result, ex016_vs_matched_control=paired)
    report = dict(status='complete_descriptive_only', plan=record(plan_path), support=support, metrics=metrics,
        curves=curve_records, model_output_metadata={label: record(output/label/'metadata.json') for label in ARMS},
        limitations=['One correlated, annotation-supported crop; not an untouched benchmark.',
            'Historical training exposure and teacher/physical independence are unresolved.',
            'Current loader removes validation from training supervision; raw S∩V alone is not certified exposure.',
            'No authenticated mesh-validity mask. Occupancy/acquisition masks do not establish geometry.',
            'No fitted thresholds, learned ensemble, calibrated confidence, text reading or accuracy authentication.'])
    write_json(directory/'metrics.json', report)
    plot_comparison(directory, arrays, labels, domains['primary'], domains['secondary'], scores, primary_curves, plan['data'])
    rows = ['| Model | ROC AUC | Average precision | BA at >= 0.5 |', '|---|---:|---:|---:|']
    for label in ARMS:
        value = metrics['primary']['models'][label]
        rows.append(f"| {label} | {value['roc_auc']:.6f} | {value['average_precision']:.6f} | {value['balanced_accuracy']:.6f} |")
    pair = metrics['primary']['ex016_vs_matched_control']
    text = '\n'.join([
        '# Fixed w029 supplied-label comparison', '',
        f"Primary common support contains {counts['primary']['pixel_count']} pixels: {counts['primary']['supplied_positive_pixels']} supplied positives and {counts['primary']['supplied_negative_pixels']} supplied negatives.", '',
        *rows, '',
        f"Ex016 minus its matched control: AUC {pair['ex016_minus_control_roc_auc']:+.6f}; average precision {pair['ex016_minus_control_average_precision']:+.6f}.", '',
        'These are descriptive supplied-label results on one fixed crop. The higher numerical score is not evidence of authentic text or generalization. No checkpoint, crop, threshold or depth window was fitted after seeing scores.', '',
        f"Raw supervision/validation overlap: {preflight['raw_supervision_validation_overlap_pixels']} pixels; it is retained because the pinned current loader subtracts validation during training. This does not certify historical trainer exposure or teacher/physical independence.", '',
        'Acquisition support and selected-depth nonzero occupancy remain separate. Zero profiles are retained when actually observed and evaluated. No authenticated geometry-validity mask is available. Full-mask secondary results, occupancy strata, grouped ROC/PR curves and raw float maps are preserved in the linked JSON/NumPy artifacts.', '',
        'This stage does not choose a new model or authorize target inference, training, a reading, or a GPU job.', ''])
    (directory/'RESULTS.md').write_text(text)
    final_plan, _ = load_plan(plan_path, execute=True)
    require((output/'frozen-plan.json').read_bytes() == plan_path.read_bytes(), 'Frozen plan changed during analysis')
    load_data(final_plan)
    for artifact in final_plan['models']:
        helper.verify_artifact(artifact['path'], artifact)
        verify_record(metadata[artifact['label']]['raw_scores'])
        verify_record(metadata[artifact['label']]['evaluated_mask'])
    write_json(directory/'final-integrity.json', dict(status='passed', plan=record(plan_path),
        code=plan['code'], inputs=plan['inputs'], model_contract_helper=plan['model_contract_helper'],
        model_contract_audit=plan['model_contract_audit'], data_contract=plan['data_contract'],
        artifacts=plan['models'], verified_after_analysis=True))
    return dict(metrics=record(directory/'metrics.json'), support=record(directory/'common-support.json'),
                report=record(directory/'RESULTS.md'), integrity=record(directory/'final-integrity.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(analyze(args.plan, args.output_dir))


if __name__ == '__main__':
    main()
