"""AST-extracted synthetic contracts only; never import Torch or model code."""
import ast
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np

HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE.parent / 'implementation'


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    namespace = dict(np=np, hashlib=hashlib, json=json, require=require)
    extract(IMPLEMENTATION/'run_worker.py', ['tree_hash'], namespace)
    extract(IMPLEMENTATION/'control_common.py', ['average', 'eligible'], namespace)
    fake_torch = types.SimpleNamespace(Tensor=type('UnusedSyntheticTensor', (), {}))
    rng = np.random.RandomState(31).get_state()
    first = namespace['tree_hash'](rng, fake_torch)
    copied = (rng[0], rng[1].copy(), *rng[2:])
    assert namespace['tree_hash'](copied, fake_torch) == first
    changed = (rng[0], rng[1].copy(), *rng[2:])
    changed[1][11] ^= np.uint32(1)
    assert namespace['tree_hash'](changed, fake_torch) != first
    assert namespace['tree_hash'](np.array([0.], dtype=np.float32), fake_torch) != namespace['tree_hash'](np.array([-0.], dtype=np.float32), fake_torch)
    assert namespace['tree_hash'](np.array([1], dtype=np.uint32), fake_torch) != namespace['tree_hash'](np.array([1], dtype=np.int32), fake_torch)
    rows = [dict(base=float(x), full=float(x)+.25) for x in ([1e16, 1., -1e16, 3.] * 8)]
    means = namespace['average'](rows)
    assert means['base'] == math.fsum(row['base'] for row in rows)/32 == 1.
    cases = [(-1e-4, -1e-4, True), (np.nextafter(-1e-4, math.inf), -1e-4, False),
             (-1e-4, np.nextafter(-1e-4, math.inf), False), (-.01, 0., False), (0., -.01, False)]
    for base, full, expected in cases:
        assert bool(namespace['eligible'](dict(base=0., full=0.), dict(base=base, full=full))) is expected
    result = dict(status='PASS', numpy_rng_hash_cases=2, bitwise_hash_cases=2,
                  cancellation_sensitive_mean_cases=1, joint_gate_cases=5,
                  source_sha256={name: hashlib.sha256((IMPLEMENTATION/name).read_bytes()).hexdigest()
                      for name in ['run_worker.py', 'control_common.py']},
                  actual_array_reads=0, model_calls=0, torch_imports=0)
    target = HERE/'source-kernel-fixtures.json'
    assert not target.exists(), 'Preserve prior fixture receipt'
    target.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
