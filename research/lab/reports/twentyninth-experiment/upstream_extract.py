"""Extract reviewed definitions/assignments only; never import the upstream modules."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / 'acquisition-v1/SOURCE_RECEIPT.json').read_text())['sources']
EXTRACTIONS = []


def extract(filename, names=(), assignments=()):
    path = HERE / 'acquisition-v1' / (filename + '.source')
    record = next(r for r in PINS if r['path'] == str(path))
    raw = path.read_bytes()
    assert len(raw) == record['size_bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    tree = ast.parse(raw, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    def target_name(target):
        while isinstance(target, ast.Subscript):
            target = target.value
        return target.id if isinstance(target, ast.Name) else None
    for name in assignments:
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and target_name(n.targets[0]) == name]
        # band_mean has one allocation and one subscript assignment; choose the latter explicitly.
        if name == 'band_mean':
            matches = [n for n in matches if isinstance(n.targets[0], ast.Subscript)]
        assert len(matches) == 1, (name, len(matches))
        nodes.append(matches[0])
    EXTRACTIONS.append(dict(source=record, nodes=[dict(start=n.lineno, end=n.end_lineno,
                             kind=type(n).__name__) for n in nodes]))
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec')


def scorer_namespace():
    ns = dict(np=np, Path=Path, zarr=zarr, NBINS=1024)
    exec(extract('score_3d.py', names=('Volume', 'tile_hist', 'auc_from_hist', 'ap_from_hist')), ns)
    return ns


def sample_assignments(ns):
    exec(extract('step1_ink_depth_profile.py', names=('sample',)), ns)
    exec(extract('step2_surface_aligned_profile.py', assignments=('idx', 'is_ink')), ns)


def score_assignments(ns, names):
    exec(extract('score_3d.py', assignments=names), ns)
