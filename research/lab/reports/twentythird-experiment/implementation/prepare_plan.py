"""Bind existing metadata and freeze before any native scientific output."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from common import HERE,PHASE,LAB,pin,write_json


def main():
    previous=LAB/'reports/twentysecond-experiment'
    sources=dict(protocol=pin(PHASE/'PROTOCOL.md'),
        input_contract=pin(PHASE/'input-audit/INPUT_CONTRACT.json'),
        input_verification=pin(PHASE/'input-audit/verification.json'),
        input_seal=pin(PHASE/'input-audit/RETENTION_MANIFEST.json'),
        support_contract=pin(PHASE/'input-audit/SUPPORT_CONTRACT.json'),
        phase22_plan=pin(previous/'model-parity/PLAN.json'),
        phase22_factory_helper=pin(previous/'model-parity/parity_sources.py'),
        phase22_state_source=pin(previous/'model-parity/run_parity.py'),
        phase22_model_seal=pin(previous/'model-parity/RETENTION_MANIFEST.json'),
        phase22_independent_verification=pin(previous/'parity-review/verification-v1.json'),
        phase22_independent_seal=pin(previous/'parity-review/RETENTION_MANIFEST.json'),
        inference_adapter=pin(LAB/'run_ink_reference.py'),
        normalization=pin(LAB/'vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py'))
    contract=json.loads(Path(sources['input_contract']['path']).read_text())
    path=Path(sources['phase22_state_source']['path']);text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='state_receipt')
    extraction=dict(source=sources['phase22_state_source'],function='state_receipt',first_line=node.lineno,last_line=node.end_lineno,
        source_segment_sha256=hashlib.sha256(''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno]).encode()).hexdigest(),
        method='Compile unchanged AST node with torch/hashlib/json/require bindings; no phase22 worker or checkpoint code executes during import')
    write_json(HERE/'STATE_EXTRACTION.json',extraction);sources['state_extraction']=pin(HERE/'STATE_EXTRACTION.json')
    code={p.name:pin(p) for p in sorted(HERE.glob('*.py'))}
    freeze=dict(status='FROZEN_PRE_NATIVE_OUTPUT',code=code,protocol=sources['protocol'],
        synthetic_checks=pin(HERE/'synthetic-checks.json'),no_further_edits=True,
        actual_model_constructions=0,actual_checkpoint_loads=0,scientific_forwards=0)
    write_json(HERE/'CODE_FREEZE.json',freeze)
    plan=dict(kind='phase23-native-reused-development-comparison-v1',sources=sources,code=code,
        code_freeze=pin(HERE/'CODE_FREEZE.json'),case_ids=[c['id'] for c in contract['cases']],seed=20261009,
        runtime=dict(python='3.12.14',numpy='2.5.3',torch='2.12.0+cpu',safetensors='0.8.0'),
        expected_model_constructions=1,expected_strict_loads=1,expected_full_model_calls=150,
        limits=dict(worker_wall_seconds=180,worker_cpu_seconds=360,external_wall_seconds=240,
            rss_bytes=4*1024**3,address_space_bytes=8*1024**3,phase_output_bytes=256*1024**2),
        input_policy='Exact contract primary mask and all150 normalized hashes before their calls; no intersection or skipped support',
        analysis=dict(primary='raw ROC AUC and tied-group AP per six contexts and separate equal-context Paris1/Paris2 means',
            diagnostics_both_models=['uint8 rank metrics','decoded rank metrics','unique scores per representation','decoded0/1 fractions'],
            paired=['mean_signed','mean_absolute','p95_absolute_linear','max_absolute'],
            metric_tolerance=1e-12,threshold_fits=0,threshold_classifications=0,performance_gate=None,automatic_promotion=False),
        no_retry=True,network_requests=0,training_steps=0,
        execution_requires='Exact root GO plan and code_freeze objects; frozen independent technical review before native execution')
    write_json(HERE/'PLAN.json',plan)
    print(json.dumps(dict(plan=pin(HERE/'PLAN.json'),code_freeze=pin(HERE/'CODE_FREEZE.json'))))


if __name__=='__main__':main()
