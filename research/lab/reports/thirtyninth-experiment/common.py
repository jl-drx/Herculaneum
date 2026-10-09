"""Phase39 file contracts; no model loading, source acquisition, or training."""
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL_LAB = Path('/workspace/herculaneum-lab')
LAB = HERE.parent.parent
KEYS = ('average_precision', 'roc_auc')
ROLES = ('seed42', 'A_control-512', 'C_coverage-512', 'midpoint')

def now():
    return datetime.now(timezone.utc).isoformat()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)

def resolve(original):
    return LAB / Path(original).relative_to(ORIGINAL_LAB)

def verify(item):
    path = resolve(item['path'])
    actual = pin(path)
    require(actual['sha256'] == item['sha256'] and actual['size_bytes'] == item['size_bytes'], 'Input identity: ' + item['path'])
    return path

def save(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def load_array(item):
    a = np.load(verify(item), allow_pickle=False)
    require(a.shape == (384,384), 'Fixed crop shape')
    return a

def midpoint(a, b):
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape, 'Retained FP32 inputs')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Finite inputs')
    return .5*a.astype(np.float64) + .5*b.astype(np.float64)

def summarise(contexts, cases):
    parents = {parent: {key: math.fsum(contexts[c['id']][key] for c in cases if c['parent']==parent)/sum(c['parent']==parent for c in cases) for key in KEYS} for parent in ('Paris1','Paris2','PHerc0814')}
    return dict(contexts=contexts, parents=parents, equal_parent_Paris={key:(parents['Paris1'][key]+parents['Paris2'][key])/2 for key in KEYS})

def decisions(roles):
    baseline = roles['seed42']
    deltas = {role:{cid:{key:row[key]-baseline['contexts'][cid][key] for key in KEYS} for cid,row in roles[role]['contexts'].items()} for role in ROLES}
    worst = {role:{key:min(row[key] for row in deltas[role].values()) for key in KEYS} for role in ROLES}
    gains = {ref:roles['midpoint']['equal_parent_Paris']['average_precision']-roles[ref]['equal_parent_Paris']['average_precision'] for ref in ('seed42','A_control-512')}
    failures = [dict(context=cid,metric=key,delta=value) for cid,row in deltas['midpoint'].items() for key,value in row.items() if value < -.02]
    screen = dict(Paris_AP_gains=gains, minimum_gain_each=.01, regional_failures=failures, passes=all(v>=.01 for v in gains.values()) and not failures)
    relation = dict(Paris_AP_no_lower=roles['midpoint']['equal_parent_Paris']['average_precision']>=roles['C_coverage-512']['equal_parent_Paris']['average_precision'], worst_AP_delta_no_lower=worst['midpoint']['average_precision']>=worst['C_coverage-512']['average_precision'], worst_AUROC_delta_no_lower=worst['midpoint']['roc_auc']>=worst['C_coverage-512']['roc_auc'])
    relation['passes'] = all(relation.values())
    return dict(region_deltas_vs_seed42=deltas, worst_deltas_vs_seed42=worst, advancement_screen=screen, descriptive_relation=relation)

def metric_module(path):
    spec = importlib.util.spec_from_file_location('phase39_binary_metrics', path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def frozen_plan():
    go = json.loads((HERE/'GO.json').read_text())
    for item in go['pins']:
        verify(item)
    return json.loads((HERE/'PLAN.json').read_text())
