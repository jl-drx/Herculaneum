"""Metadata-only binding and freeze for the paired normalized-depth intervention."""
from __future__ import annotations
import json
from pathlib import Path
from common import HERE,PHASE,LAB,pin,write_json


def main():
    prior22=LAB/'reports/twentysecond-experiment';prior23=LAB/'reports/twentythird-experiment'
    contract=json.loads((PHASE/'input-audit/MODEL_INPUT_CONTRACT.json').read_text())
    sources=dict(protocol=pin(PHASE/'PROTOCOL.md'),input_contract=pin(PHASE/'input-audit/MODEL_INPUT_CONTRACT.json'),
        input_verification=pin(PHASE/'input-audit/MODEL_INPUT_DERIVATION.json'),
        input_diagnostic_verification=pin(PHASE/'input-audit/verification.json'),
        input_seal=pin(PHASE/'input-audit/RETENTION_MANIFEST.json'),
        phase22_plan=pin(prior22/'model-parity/PLAN.json'),
        phase22_factory_helper=pin(prior22/'model-parity/parity_sources.py'),
        phase22_state_source=pin(prior22/'model-parity/run_parity.py'),
        phase22_independent_verification=pin(prior22/'parity-review/verification-v1.json'),
        phase23_author_seal=pin(prior23/'implementation/RETENTION_MANIFEST.json'),
        phase23_independent_verification=pin(prior23/'independent-review/verification-v1.json'),
        phase23_input_contract=pin(prior23/'input-audit/INPUT_CONTRACT.json'),
        seed42_bound_config=contract['seed42_bound_configuration'],
        native_forward_load=contract['native_load_verification'],
        legacy_state_hash_source=pin(LAB/'reports/twentyfirst-experiment/allocation-implementation/depth_common.py'),
        inference_adapter=pin(LAB/'run_ink_reference.py'),
        normalization=pin(LAB/'vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py'))
    code={p.name:pin(p) for p in sorted(HERE.glob('*.py'))}
    freeze=dict(status='FROZEN_PRE_REVERSED_OUTPUT',code=code,protocol=sources['protocol'],
        synthetic_checks=pin(HERE/'synthetic-checks.json'),no_further_edits=True,
        actual_model_constructions=0,actual_checkpoint_loads=0,scientific_forwards=0)
    write_json(HERE/'CODE_FREEZE.json',freeze)
    artifact=contract['seed42_artifact_metadata_binding']
    checkpoint={k:artifact[k] for k in ('path','sha256','size_bytes')}
    plan=dict(kind='phase24-paired-normalized-depth-reversal-v1',sources=sources,code=code,
        code_freeze=pin(HERE/'CODE_FREEZE.json'),input_contract_status=contract['status'],
        case_ids=[c['id'] for c in contract['cases']],seed=20261009,
        seed42_checkpoint=checkpoint,seed42_state_sha256=contract['backbone_state_sha256'],
        runtime=dict(python='3.12.14',numpy='2.5.3',torch='2.12.0+cpu',safetensors='0.8.0'),
        execution_order=['seed42','native'],expected_model_constructions=2,expected_strict_loads=2,
        expected_full_model_calls=300,per_model_calls=150,
        intervention=dict(adapter_read_direction='forward',model_depth_permutation=list(range(16,-1,-1)),
            tensor_axis=2,verify_both_hashes_before_call=True,normalize_once=True),
        limits=dict(worker_wall_seconds=180,worker_cpu_seconds=360,external_wall_seconds=420,
            rss_bytes=4*1024**3,address_space_bytes=8*1024**3,phase_output_bytes=256*1024**2),
        analysis=dict(primary='Raw AUC/AP for both models and both directions per six contexts and separate equal-context parents',
            changes='reverse-minus-forward per model; difference_of_changes=native_change-seed42_change',
            drift=['mean_signed','mean_absolute','p95_absolute_linear','max_absolute'],
            metric_tolerance=1e-12,threshold_fits=0,threshold_classifications=0,quantized_metrics=False,
            performance_gate=None,automatic_promotion=False,direction_selection=False),
        no_retry=True,network_requests=0,training_steps=0,
        execution_requires='Exact root GO plan/code_freeze objects and frozen independent technical review')
    write_json(HERE/'PLAN.json',plan)
    print(json.dumps(dict(plan=pin(HERE/'PLAN.json'),code_freeze=pin(HERE/'CODE_FREEZE.json'))))


if __name__=='__main__':main()
