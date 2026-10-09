"""Freeze source-bound Stage2 plan; inspect metadata only, never checkpoint values."""
from __future__ import annotations
import json
from pathlib import Path
from parity_sources import HERE, PHASE, LAB, INPUT, pin, extraction_records, verify


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def main():
    sources={
        'protocol': pin(PHASE/'SYNTHETIC_PARITY_PROTOCOL.md'),
        'qualification_protocol':pin(PHASE/'CHECKPOINT_QUALIFICATION_PROTOCOL.md'),
        'historical_closure':pin(INPUT/'HISTORICAL_FACTORY_CLOSURE.json'),
        'historical_sources_seal':pin(INPUT/'RETENTION_MANIFEST.json'),
        'input_contract':pin(INPUT/'INPUT_CONTRACT.json'),
        'reference_schema':pin(LAB/'reports/twentyfirst-experiment/native-checkpoint-contract/REFERENCE_SCHEMA.json'),
        'current_vendor_manifest':pin(LAB/'vendor/villa-ink-baseline/source-manifest.json'),
        'baseline_adapter':pin(LAB/'run_ink_baseline.py'),
    }
    closure=json.loads(verify(sources['historical_closure']).read_text())
    source=closure['historical_inference_source_for_reviewed_pure_extraction']
    write(HERE/'EXTRACTION_PROVENANCE.json',dict(source=source,
        definitions=extraction_records(source),method='Unmodified AST nodes; injected safe standard/library globals; future annotations',
        excluded=['all upstream imports','module-level multiprocessing start method','unsafe checkpoint loader',
                  'CLI/main','Zarr/network open','TIFF writer','TTA'],
        deviations=['In-memory NumPy source replaces Zarr object; _array populated before exact reader access',
                    'batch_size1,num_workers0 DataLoader; tqdm replaced with identity iterator',
                    'Whole-canvas synthetic serialization tile; no compressed TIFF file written',
                    'RecordingModel preserves pre-sigmoid logits and returns exact ink tensor directly; no extra model calls']))
    sources['extraction_provenance']=pin(HERE/'EXTRACTION_PROVENANCE.json')
    cases=[]
    for name,depth,interval,reverse,size,zero,forwards in [
        ('native-default',28,[0,28],False,128,False,1),
        ('native-window-0-20',28,[0,20],False,128,False,1),
        ('native-window-3-23',28,[3,23],False,128,False,1),
        ('native-window-5-25',28,[5,25],False,128,False,1),
        ('native-window-8-27',28,[8,27],False,128,False,1),
        ('aligned-default',21,[0,21],False,128,False,1),
        ('native-reverse',28,[0,28],True,128,False,1),
        ('native-tiled',28,[0,28],False,192,False,4),
        ('zero-skip',28,[0,28],False,128,True,0),
    ]:
        cases.append(dict(name=name,shape=[depth,size,size],interval=interval,reverse=reverse,zero=zero,forwards=forwards))
    code={p.name:pin(p) for p in sorted(HERE.glob('*.py'))}
    freeze=dict(status='FROZEN_PRE_EXECUTION',protocol=sources['protocol'],code=code,
        toy_tests=pin(HERE/'toy-tests-v3.json'),no_further_edits=True,
        real_factory_constructions=0,real_checkpoint_loads=0,real_model_forwards=0)
    write(HERE/'CODE_FREEZE.json',freeze)
    # Identity already proven by independent Stage1 inspectors; no checkpoint bytes read here.
    artifact=dict(path=str(PHASE/'checkpoint-qualification/run-v1/dense_native_016000.safetensors'),
        size_bytes=272767904,sha256='d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d')
    plan=dict(kind='phase22-native-cpu-synthetic-parity-v1',sources=sources,code=code,
        code_freeze=pin(HERE/'CODE_FREEZE.json'),
        stage1=dict(artifact=artifact,author_receipt=pin(PHASE/'checkpoint-qualification/run-v1/qualification.json'),
            independent_receipt=pin(PHASE/'checkpoint-review/verification-v1.json')),
        runtime=dict(python='3.12.14',numpy='2.5.3',torch='2.12.0+cpu',safetensors='0.8.0'),
        seed=20261009,cases=cases,forwards_per_worker=11,
        input_formula='uint8(1+((17*z+3*y+5*x+(y*x)%29+(z*y)%23)%251))',
        acceptance=dict(exact=['indices','tile_schedule','normalized_tensor_bytes','state_bytes','evaluated_masks'],
            logits=dict(atol=1e-5,rtol=1e-5,reference='historical'),
            probabilities_and_blends=dict(atol=2e-6,rtol=2e-6,reference='historical'),
            quantization='each implementation exact against its own saved raw score; cross-factory differences diagnostic'),
        limits=dict(worker_wall_seconds=180,worker_external_seconds=180,worker_cpu_seconds=360,
            controller_wall_seconds=400,rss_bytes=4*1024**3,address_space_bytes=8*1024**3,
            stage_output_bytes=128*1024**2,phase_output_bytes=768*1024**2,threads=4),
        execution_requires='Root GO with exact plan and CODE_FREEZE pin plus Stage1 receipts PASS',
        no_retry=True,scientific_inputs=0,network_requests=0,training_steps=0)
    write(HERE/'PLAN.json',plan)
    print(json.dumps(dict(plan=pin(HERE/'PLAN.json'),code_freeze=pin(HERE/'CODE_FREEZE.json'))))


if __name__=='__main__':main()
