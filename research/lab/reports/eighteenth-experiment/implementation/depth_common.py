"""Explicit file pins, state receipts and resource gates for phase18."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import resource
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUNTIME = dict(python='3.12.14', numpy='2.5.3', torch='2.12.0+cpu')


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def verify(pin):
    require(record(pin['path']) == pin, 'File pin changed: ' + pin['path'])
    return Path(pin['path'])


def read_json(pin):
    return json.loads(verify(pin).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(pin, name):
    spec = importlib.util.spec_from_file_location(name, verify(pin))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def state_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(array.shape), str(array.dtype)], separators=(',', ':')).encode() + b'\0')
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def tree_hash(value):
    import torch
    digest = hashlib.sha256()
    def frame(tag, data=b''):
        digest.update(tag.encode() + b'\0' + len(data).to_bytes(8, 'big') + data)
    def visit(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous()
            frame('tensor_metadata', json.dumps([list(a.shape), str(a.dtype)]).encode())
            frame('tensor_data', a.numpy().tobytes())
        elif isinstance(v, np.ndarray):
            frame('numpy_metadata', json.dumps([list(v.shape), str(v.dtype)]).encode())
            frame('numpy_data', v.tobytes(order='C'))
        elif isinstance(v, dict):
            frame('dict', str(len(v)).encode())
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list)):
            frame(type(v).__name__, str(len(v)).encode())
            for item in v:
                visit(item)
        else:
            frame(type(v).__name__, json.dumps(v, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def isolation_state(adapter, optimizer=None):
    import torch
    return dict(adapter=state_hash(adapter.state_dict()),
                optimizer=tree_hash(optimizer.state_dict()) if optimizer else None,
                gradients=tree_hash({name:p.grad for name,p in adapter.named_parameters()}),
                modes=[part.training for part in adapter.modules()],
                torch_rng=tree_hash(torch.random.get_rng_state()),
                numpy_rng=tree_hash(np.random.get_state()), python_rng=tree_hash(random.getstate()))


def equal_batch_mean(rows):
    require(len(rows) == 32 and all(set(row) == set(rows[0]) for row in rows), 'Exactly32 aligned risk rows required')
    require(all(math.isfinite(v) for row in rows for v in row.values()), 'Nonfinite risk')
    return {key:math.fsum(row[key] for row in rows) / 32 for key in rows[0]}


def rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def output_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())


def load_plan(path, execute=False):
    path = Path(path).resolve(strict=True)
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'phase18-depth-residual-v1' and plan['runtime'] == RUNTIME, 'Plan/runtime differs')
    require(plan['arms'] == ['raw_control', 'explicit_depth'], 'Arm order differs')
    require(plan['training']['steps'] == 32 and plan['training']['fixed_updates'] == [1,8,16,32], 'Frozen schedule differs')
    verify(plan['protocol'])
    config = read_json(plan['configuration'])
    require(all(plan[key] == config[key] for key in ['architecture','training','limits','gates']), 'Frozen configuration differs')
    for name, pin in plan['code'].items():
        require(verify(pin) == HERE/name, 'Executable source path differs')
    require(set(plan['code']) == {p.name for p in HERE.glob('*.py')}, 'Executable source set differs')
    for pin in plan['source_code'].values():
        verify(pin)
    if execute:
        require(plan['ready_for_execution'] is True, 'Execution not authorized')
        freeze = read_json(plan['code_freeze'])
        go = json.loads(Path(plan['source_go_path']).read_text())
        require(freeze['no_further_edits'] is True and freeze['code'] == plan['code']
                and freeze['protocol'] == plan['protocol'], 'Code/protocol freeze differs')
        require(go['status'] == 'GO' and go['code_freeze'] == plan['code_freeze']
                and go['protocol'] == plan['protocol'] and go['plan'] == record(path),
                'Root scientific GO missing or does not bind this exact plan')
    return plan, path
