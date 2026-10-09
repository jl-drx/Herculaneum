#!/usr/bin/env python3
"""Lock four thresholds from completed sixth-stage development maps; no inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np

from operating_point import choose_development_threshold, operating_metrics

SIXTH_COMMON = Path('/workspace/herculaneum-lab/reports/sixth-experiment/benchmark/comparison_common.py')
SIXTH_COMMON_SHA = 'dd15e4870c1d980a62fa1eabe9ffd26657c9b6b9c091c5ac6a180c9b7333808a'


def load_sixth_common():
    import hashlib
    with SIXTH_COMMON.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != SIXTH_COMMON_SHA:
        raise ValueError('Verified sixth-stage contract source changed')
    spec = importlib.util.spec_from_file_location('verified_sixth_comparison_common', SIXTH_COMMON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lock(sixth_plan, sixth_run, output):
    common = load_sixth_common()
    plan, sixth_plan = common.load_plan(sixth_plan, execute=True)
    arrays, labels, primary, _, _ = common.load_data(plan)
    sixth_run = Path(sixth_run).resolve(strict=True)
    common.require((sixth_run/'frozen-plan.json').read_bytes() == sixth_plan.read_bytes(), 'Sixth run plan differs')
    support_path = sixth_run/'analysis/common-support.json'
    support = json.loads(support_path.read_text())
    primary_path = common.verify_record(support['mask_records']['primary'])
    primary_saved = np.load(primary_path, allow_pickle=False)
    common.require(primary_saved.dtype == np.bool_ and primary_saved.shape == labels.shape, 'Invalid development primary mask')
    common_model_support = np.ones(labels.shape, dtype=bool)
    maps, output_records = {}, {}
    helper = common.import_helper(plan)
    for artifact in plan['models']:
        role = artifact['label']
        helper.verify_artifact(artifact['path'], artifact)
        metadata_path = sixth_run/role/'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        common.require(metadata['plan']['sha256'] == common.record(sixth_plan)['sha256']
            and metadata['model']['sha256'] == artifact['sha256'] and metadata['model']['strict_state_match'], 'Development output/model contract mismatch')
        score_path = common.verify_record(metadata['raw_scores'])
        evaluated_path = common.verify_record(metadata['evaluated_mask'])
        scores = np.load(score_path, allow_pickle=False)
        evaluated = np.load(evaluated_path, allow_pickle=False)
        common.require(scores.dtype == np.float32 and scores.shape == labels.shape
            and evaluated.dtype == np.bool_ and evaluated.shape == labels.shape, 'Invalid development map contract')
        common.require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'Invalid development scores')
        common_model_support &= evaluated & np.isfinite(scores)
        maps[role] = scores
        output_records[role] = dict(scores=common.record(score_path), evaluated_mask=common.record(evaluated_path),
            metadata=common.record(metadata_path), checkpoint=artifact)
    primary &= common_model_support
    common.require(np.array_equal(primary, primary_saved), 'Development common mask differs from reconstructed sixth-stage domain')
    counts = common.class_counts(labels, primary)
    common.require(common.sufficient(counts), 'Insufficient development primary classes')
    thresholds = {}
    for role in plan['model_order']:
        threshold = choose_development_threshold(maps[role][primary & ~labels])
        thresholds[role] = dict(**threshold, development_operating_metrics=operating_metrics(
            labels[primary], maps[role][primary], threshold['threshold']))
    output = Path(output).resolve()
    common.require(not output.exists(), 'Threshold lock already exists; never overwrite it')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, status='locked_development_thresholds_no_inference',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        development_role='Previously inspected w029 is used development data, not a reserved or physically heldout region.',
        development_case='pherc1667-w029', development_physical_group='PHerc1667',
        development_domain='Sixth-stage fixed64-margin primary common support only; no reserved labels or scores used.',
        development_counts=counts, thresholds=thresholds, model_order=plan['model_order'],
        sixth_plan=common.record(sixth_plan), primary_mask=common.record(primary_path),
        source_inputs=plan['inputs'], source_provenance=plan['data_contract'],
        models_and_maps=output_records, source_support=common.record(support_path),
        code=dict(operating_point=common.record(Path(__file__).parent/'operating_point.py'),
                  lock_driver=common.record(Path(__file__)), sixth_common=common.record(SIXTH_COMMON)),
        rule='Each model independently uses floor(n_negative/20) maximum FP; float32 nextafter excludes all ties at kth boundary. Reserved thresholds are never retuned.',
        inference_calls=0, limitation='Empirical development specificity; no calibrated confidence, independent truth or guarantee of reserved specificity.')
    common.load_plan(sixth_plan, execute=True)
    common.load_data(plan)
    for item in output_records.values():
        for key in ['scores', 'evaluated_mask', 'metadata']:
            common.verify_record(item[key])
    common.write_json(output, manifest)
    return common.record(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sixth-plan', required=True, type=Path)
    parser.add_argument('--sixth-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(lock(args.sixth_plan, args.sixth_run, args.output)))


if __name__ == '__main__':
    main()
