"""Pin qualified code, inputs, checkpoints and environment before inference."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import shutil
import subprocess
import cv2
import numpy as np
import torch
import runner as r


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); run = a.run; here = Path(__file__).resolve().parent
    assert not (run/'GO.json').exists() and not (run/'jobs').exists(), 'Already frozen or inference begun'
    qualification = r.read(run/'QUALIFICATION.json'); assert qualification['status'] == 'PASS'
    for row in qualification['code']:
        assert r.pin(row['path']) == row
    assert r.sha(a.inputs/'SELECTION.json') == 'bb19d53736cf3491e8ee7847a89d0632b6cffd441346730d4855e485cd17615c'
    assert r.sha(a.inputs/'PATCHES_COMPLETE.json') == '5364e1e9aa2f537dd69ba26179b4d3ff82304fde69f5999ff316f661c4925a7e'
    assert r.read(a.inputs/'VERIFIED_INPUTS.json')['status'] == 'PASS'
    code = run/'implementation'; code.mkdir(parents=True)
    for name in ['runner.py', 'qualify.py', 'freeze.py', 'backup.py', 'PROTOCOL.md']:
        shutil.copy2(here/name, code/name)
    paths = [a.inputs/name for name in ['FREEZE.json', 'SUPERVISION_GO.json', 'SELECTION.json',
             'CT_PLAN.json', 'CT_COMPLETE.json', 'INPUTS_COMPLETE.json', 'PATCHES_COMPLETE.json', 'VERIFIED_INPUTS.json']]
    paths += sorted((a.inputs/'cores').glob('*.npz'))
    source = a.repo/'research/lab/reports/fortysixth-experiment/runtime-source'
    runtime = sorted(p for p in source.rglob('*') if p.suffix in ('.py', '.json'))
    runtime.append(a.repo/'research/lab/reports/fortyfifth-experiment/source-evidence/hecate-hecate.py')
    pins = r.read(a.repo/'research/in-progress/native-common-support-20261012/MODEL_PINS.json')['models']
    reader = Path('/workspace/herculaneum-lab/data/models/reader-v2-20261012/reader-v2-step040000.raw')
    assert r.sha(reader) == pins[0]['checkpoint_sha256'] and reader.stat().st_size == pins[0]['bytes']
    models = [dict(name='reader_v2', checkpoint=r.pin(reader))]
    for name, row in zip(['dnative', 'seed42', 'seed43', 'hecate'], pins[1:]):
        path = Path(row['checkpoint']['path']); actual = r.pin(path)
        assert actual['sha256'] == row['checkpoint']['sha256'] and actual['bytes'] == row['checkpoint']['size_bytes']
        models.append(dict(name=name, checkpoint=actual))
    packages = {dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()}
    environment = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        opencv=cv2.__version__, platform=platform.platform(), packages=packages)
    bundle = r.read(a.repo/'research/in-progress/native-common-support-20261012/BUNDLE.json')
    go = dict(schema_version=1, status='FROZEN_BEFORE_TRAINED_MODEL_INFERENCE', utc=r.now(),
        plan=dict(study='native-common-support-five-model-comparison-20261012', model_order=[x['name'] for x in models],
                  expected_jobs=20, expected_core_maps=120, expected_unique_forwards=1698,
                  backup_interval_seconds=300, metric_lock='No metrics until all jobs complete; implementation/PROTOCOL.md'),
        models=models, environment=environment, input_pins=[r.pin(x, a.inputs) for x in paths],
        runtime_pins=[r.pin(x, a.repo) for x in runtime],
        code_pins=[r.pin(x, run) for x in sorted(code.iterdir())]+[r.pin(run/'QUALIFICATION.json', run)],
        source_archives=[dict(path='research/workspace/bundles/'+bundle['archive'], bytes=bundle['bytes'],
                             sha256=bundle['sha256'], commit='eb5947970e0fc5279fe36558daf6a25cfd3dfd2d',
                             purpose='Frozen native common-support inputs; model weights separate')],
        base_commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
        exposure='Adaptive supervision-only development panel; transferred/model-reviewed reference; upstream exposure unresolved',
        reserved_families=['PHerc0175 A', 'PHerc0175 B', 'PHerc1203'], paid_compute_authorized=False)
    r.atomic(run/'GO.json', go)
    for name in ['GO.json', 'QUALIFICATION.json']:
        shutil.copy2(run/name, here/name)
    print(r.sha(run/'GO.json'))


if __name__ == '__main__':
    main()
