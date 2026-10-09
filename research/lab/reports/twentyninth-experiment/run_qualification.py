"""One frozen offline qualification run; refuses reuse of the output directory."""
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import resource
import signal
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent


def record(path):
    b = path.read_bytes()
    return dict(path=str(path.resolve()), size_bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def main():
    signal.alarm(120)
    freeze = json.loads((HERE / 'SOURCE_FREEZE.json').read_text())
    for expected in freeze['files']:
        assert record(Path(expected['path'])) == expected, 'Frozen qualification source changed'
    run = HERE / 'run-v1'; run.mkdir(exist_ok=False)
    (run / 'fixtures').mkdir()
    denied = []
    def block_network_and_process(event, args):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
            denied.append(event)
            raise RuntimeError('Offline qualification forbids network/process launch')
    sys.addaudithook(block_network_and_process)
    start = time.perf_counter()
    import test_reference_metrics as tests
    import upstream_extract
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    (run / 'TEST_LOG.txt').write_text(stream.getvalue())
    (run / 'OBSERVATIONS.json').write_text(json.dumps(tests.OBSERVATIONS, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (run / 'EXTRACTIONS.json').write_text(json.dumps(upstream_extract.EXTRACTIONS, indent=2, sort_keys=True)+'\n')
    receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors), skips=len(result.skipped),
                   elapsed_seconds=time.perf_counter()-start,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   network_or_process_attempts_denied=denied, source_freeze=record(HERE/'SOURCE_FREEZE.json'),
                   python=sys.version, dependencies={name:importlib.metadata.version(name) for name in ('numpy','zarr')},
                   actual_candidate_pixels=0, scientific_model_forwards=0, GPU_spend_GBP=0,
                   scope='Selected upstream AST paths and local bounded adapter on fixed synthetic inputs only')
    receipt['outputs'] = [record(run/n) for n in ('TEST_LOG.txt','OBSERVATIONS.json','EXTRACTIONS.json')]
    (run / 'RUN_RECEIPT.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return not result.wasSuccessful()


if __name__ == '__main__':
    raise SystemExit(main())
