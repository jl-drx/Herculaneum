"""Describe frozen scan-context runs; scores do not authenticate ink or text."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_array(path, expected=None):
    path = Path(path)
    if expected is not None and sha(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return np.load(path, allow_pickle=False)


def describe(values):
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Analysis requires nonempty finite evaluated scores.')
    return {'count': int(values.size), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'p01': float(np.percentile(values, 1)),
            'p99': float(np.percentile(values, 99)), 'maximum': float(values.max())}


def label_agreement(score, labels):
    positive = labels == 1
    negative = labels == 0
    if not np.all(positive | negative):
        raise ValueError('Only documented 0/1 labels may enter supervised agreement.')
    npos, nneg = int(positive.sum()), int(negative.sum())
    pred = score > 0.5
    tp, fp = int((pred & positive).sum()), int((pred & negative).sum())
    fn, tn = npos - tp, nneg - fp
    auc = None
    if npos and nneg:
        ranks = rankdata(score, method='average')
        auc = float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return {'ink_label_pixels': npos, 'background_label_pixels': nneg,
            'ink_label_scores': describe(score[positive]) if npos else None,
            'background_label_scores': describe(score[negative]) if nneg else None,
            'threshold_strictly_greater_than': 0.5,
            'tp_fp_fn_tn': [tp, fp, fn, tn],
            'descriptive_precision': tp / (tp + fp) if tp + fp else None,
            'descriptive_recall': tp / npos if npos else None,
            'descriptive_label_ranking_roc_auc': auc,
            'interpretation': 'Agreement with released supervised annotations on this selected, training-exposed crop; not independent physical truth or target accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Choose a fresh output directory.')
    entries = json.loads(args.manifest.read_text())['contexts']
    rows, views = [], []
    for entry in entries:
        run_dir = Path(entry['run_dir'])
        metadata_path = run_dir / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        source = metadata['source']
        surface = checked_array(source['path'], source['sha256'])
        if surface.ndim != 3 or surface.dtype != np.uint8 or not 17 <= surface.shape[0] <= 31 or surface.shape[1:] != (256, 256):
            raise ValueError('The frozen comparison requires genuine uint8[17..31,256,256] contexts.')
        if sha(source['provenance_path']) != source['provenance_sha256']:
            raise ValueError('Input provenance changed after inference.')
        outputs = metadata['outputs']
        arrays = {name: checked_array(run_dir / record['path'], record['sha256'])
                  for name, record in outputs.items()}
        for name, values in arrays.items():
            record = outputs[name]
            if list(values.shape) != record['shape_yx'] or values.dtype != np.dtype(record['dtype']) or values.shape != (256, 256):
                raise ValueError('Output shape/dtype does not match its recorded 256×256 array.')
        for name in ('validity-mask', 'seed42-evaluated-mask', 'seed43-evaluated-mask'):
            if arrays[name].dtype != np.bool_:
                raise ValueError('Validity/evaluated arrays must be boolean masks.')
        seed42, seed43 = arrays['seed42'], arrays['seed43']
        processing = metadata['processing']
        start, stop = processing['layer_window_start_stop']
        expected_window = [2, 19] if entry.get('labels_path') else [7, 24]
        if [start, stop] != expected_window or processing['depth_direction'] != 'forward':
            raise ValueError('Run differs from the frozen forward source-layer window.')
        if processing['patch_size_zyx'] != [17, 128, 128] or processing['stride_xy'] != [64, 64] or processing['device'] != 'cpu' or processing['dtype'] != 'float32':
            raise ValueError('Run differs from the common frozen inference settings.')
        if processing['normalization'] != 'Unmodified upstream normalize_robust over each complete ZYX patch: percentile clip 1/99, median/MAD*1.4826 with upstream fallbacks' or processing['blend'] != 'Nonperiodic torch Hann outer product, normalized maximum, floored at 0.001':
            raise ValueError('Run differs from the common normalization or blending method.')
        if {model['label'] for model in metadata['models']} != {'seed42', 'seed43'} or not all(model['strict_state_match'] for model in metadata['models']):
            raise ValueError('Both checkpoints must have strict state matching.')
        valid = (arrays['validity-mask'] & arrays['seed42-evaluated-mask']
                 & arrays['seed43-evaluated-mask'] & np.any(surface[start:stop] != 0, axis=0))
        center = np.zeros((256, 256), dtype=bool)
        center[64:192, 64:192] = True
        analysis = valid & center
        labels, supervision = None, None
        if entry.get('labels_path'):
            labels = checked_array(entry['labels_path'], entry['labels_sha256'])
            supervision = checked_array(entry['supervision_path'], entry['supervision_sha256'])
            if labels.shape != (256, 256) or supervision.shape != (256, 256) or supervision.dtype != np.bool_:
                raise ValueError('Labels and boolean supervision must align with the context.')
            analysis &= supervision
            if not np.all(np.isin(labels[analysis], [0, 1])):
                raise ValueError('Undocumented or ignore values entered the analysis mask.')
        if not analysis.any():
            raise ValueError('No pixels satisfy the frozen analysis mask.')
        row = {'name': entry['name'], 'role': entry['role'],
               'run_metadata_path': str(metadata_path), 'run_metadata_sha256': sha(metadata_path),
               'source_sha256': source['sha256'], 'source_shape_zyx': list(surface.shape),
               'layer_window': [start, stop], 'source_provenance': source['provenance'],
               'analysis_bbox_local_xywh': [64, 64, 128, 128],
               'analysis_pixels': int(analysis.sum()), 'excluded_central_pixels': int(center.sum() - analysis.sum()),
               'seed42': describe(seed42[analysis]), 'seed43': describe(seed43[analysis]),
               'shared_scores_gt_0_5_count': int(((seed42 > .5) & (seed43 > .5) & analysis).sum()),
               'shared_scores_gt_0_5_fraction': float(((seed42 > .5) & (seed43 > .5))[analysis].mean()),
               'seed_mean_absolute_difference': float(np.abs(seed42[analysis] - seed43[analysis]).mean())}
        row['seed_pearson'] = float(np.corrcoef(seed42[analysis], seed43[analysis])[0, 1]) if min(seed42[analysis].std(), seed43[analysis].std()) > 0 else None
        if labels is not None:
            row['annotation_sources'] = {key: entry[key] for key in ('labels_path', 'labels_sha256', 'supervision_path', 'supervision_sha256')}
            row['annotation_agreement'] = {name: label_agreement(score[analysis], labels[analysis])
                                           for name, score in [('seed42', seed42), ('seed43', seed43)]}
        rows.append(row)
        views.append((entry['name'], surface[(start + stop - 1) // 2], seed42, seed43, valid, labels, supervision, analysis))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#dddddd')
    fig, axes = plt.subplots(len(views), 4, figsize=(16, 3.8 * len(views)), squeeze=False)
    for index, (name, observed, s42, s43, valid, labels, supervision, analysis) in enumerate(views):
        lo, hi = np.percentile(observed[valid], [1, 99])
        axes[index, 0].imshow(np.ma.array(observed, mask=~valid), cmap='gray', vmin=lo, vmax=hi)
        axes[index, 0].set_title(f'{name}\nObserved center layer, contrast 1–99%')
        for column, (title, score) in enumerate([('Seed42 raw score', s42), ('Seed43 raw score', s43), ('Minimum seed score', np.minimum(s42, s43))], start=1):
            axes[index, column].imshow(np.ma.array(score, mask=~valid), cmap=cmap, vmin=.25, vmax=.75)
            axes[index, column].set_title(title + '\nScale 0.25–0.75')
        if labels is not None:
            overlay = np.zeros((256, 256, 4))
            overlay[..., 1] = 1
            overlay[..., 3] = ((labels == 1) & supervision) * .45
            axes[index, 0].imshow(overlay)
            axes[index, 0].set_title(f'{name}\nCenter layer · contrast 1–99%\nGreen = supplied ink annotation')
        for ax in axes[index]:
            ax.add_patch(plt.Rectangle((63.5, 63.5), 128, 128, fill=False, color='cyan', linewidth=.8))
            ax.set_xticks([])
            ax.set_yticks([])
        np.save(args.output_dir / f'context-{index + 1}-analysis-mask.npy', analysis, allow_pickle=False)
        rows[index]['analysis_mask'] = {'path': f'context-{index + 1}-analysis-mask.npy', 'sha256': sha(args.output_dir / f'context-{index + 1}-analysis-mask.npy')}
    fig.suptitle('Frozen context comparison · cyan = central analysis region\nModel scores and supplied annotations do not authenticate target ink', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.output_dir / 'context-comparison.png', dpi=120)
    plt.close(fig)
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'analysis_code_sha256': sha(Path(__file__)), 'manifest_sha256': sha(args.manifest),
              'contexts': rows,
              'scientific_status': 'Descriptive detector and annotation comparison; no target transcription or translation.',
              'limits': ['Reference is training-exposed and annotations may include pseudo-labels.',
                         'Target contexts share a previously scored physical field.',
                         'Seeds share a training corpus; pixels are not independent trials.',
                         'Reference and target native spacing differ; no fitted resampling.',
                         'Source contrast 1–99% is a display adjustment; scores retain raw values.']}
    (args.output_dir / 'comparison-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps([{key: row[key] for key in ['name', 'analysis_pixels', 'shared_scores_gt_0_5_fraction']} for row in rows], indent=2))


if __name__ == '__main__':
    main()
