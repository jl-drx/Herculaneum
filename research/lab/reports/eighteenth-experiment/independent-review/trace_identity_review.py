"""Link small retained ordered parameter traces to reported semantic state hashes."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
ORDER = [('stem.weight', (8,3,3,3,3)), ('stem.bias', (8,)), ('head.weight', (1,16,3,3)), ('head.bias', (1,))]


def pin(path):
    path = Path(path).resolve(strict=True)
    if path.suffix not in {'.json', '.jsonl', '.npz', '.py'}:
        raise ValueError('Only reports, traces and this source are permitted')
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(item):
    if pin(item['path']) != item:
        raise ValueError('Pinned file changed')
    return Path(item['path'])


def state_hash(vector):
    values, offset = {}, 0
    for name, shape in ORDER:
        size = math.prod(shape)
        values[name] = vector[offset:offset+size].reshape(shape)
        offset += size
    assert offset == 801
    digest = hashlib.sha256()
    for name in sorted(values):
        array = values[name]
        digest.update(json.dumps([name, list(array.shape), 'torch.float32'], separators=(',', ':')).encode() + b'\0')
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main():
    review_pin = pin(ROOT/'training-v2/verification.json')
    assert review_pin['sha256'] == 'a8a69222639c2e7a883731f977cfc15c19d22d2401f6e9d3fe39749e78887a5f'
    review = json.loads(verify(review_pin).read_text())
    assert review['status'] == 'PASS'
    pins, reports = [review_pin], {}
    for arm, entry in review['arms'].items():
        result = json.loads(verify(entry['training_result']).read_text())
        logs = [json.loads(line) for line in verify(result['steps']).read_text().splitlines()]
        with np.load(verify(result['optimizer_traces']), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ('parameters_before', 'parameters_after', 'gradients_before_clip')}
        assert state_hash(arrays['parameters_before'][0]) == result['initial_adapter_state_sha256']
        assert state_hash(arrays['parameters_after'][-1]) == result['final_adapter_state_sha256'] == result['checkpoint_reload']['state_sha256']
        maximum = 0.
        for i, row in enumerate(logs):
            gradient = arrays['gradients_before_clip'][i].astype(np.float64)
            delta = arrays['parameters_after'][i].astype(np.float64) - arrays['parameters_before'][i].astype(np.float64)
            expected = dict(stem_gradient_l2=float(np.sqrt(np.square(gradient[:656]).sum())),
                            head_gradient_l2=float(np.sqrt(np.square(gradient[656:]).sum())),
                            parameter_update_l2=float(np.sqrt(np.square(delta).sum())))
            for key, value in expected.items():
                error = abs(row[key] - value)
                assert error <= 1e-12, (arm, i, key, error)
                maximum = max(maximum, error)
        pins += [entry['training_result'], result['steps'], result['optimizer_traces']]
        reports[arm] = dict(initial_semantic_state_sha256=result['initial_adapter_state_sha256'],
                           final_semantic_state_sha256=result['final_adapter_state_sha256'],
                           state_trace_linkage_exact=True, norm_comparisons=96, maximum_norm_error=maximum)
    for item in pins:
        verify(item)
    assert 'torch' not in sys.modules
    result = dict(status='PASS', arms=reports, inputs=pins, source=pin(__file__), checkpoint_decodes=0,
                  model_calls=0, training_replays=0,
                  limitation='Checkpoint bytes were not decoded; retained trace state hashes agree with strict-reload receipts')
    with (ROOT/'trace-identity-verification.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True);stream.write('\n')
    print(json.dumps(dict(status='PASS', receipt=pin(ROOT/'trace-identity-verification.json')), sort_keys=True))


if __name__ == '__main__':
    main()
