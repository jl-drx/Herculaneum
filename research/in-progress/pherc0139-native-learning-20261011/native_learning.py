"""Frozen, CPU-only native annotation learning with physical-context exclusions.

All paths are explicit and relocatable. No network calls, pretrained weights or
held-out predictions enter training. See PROTOCOL.md for scientific limitations.
"""
import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn
import torch.nn.functional as F
import zarr


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pin(path, root):
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def verified(root, row):
    path = root / row['path']
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes root')
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Changed input/output: ' + str(path))
    return path


def setup():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def environment():
    return dict(python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, zarr=zarr.__version__,
                threads=torch.get_num_threads(), machine=platform.machine(),
                cuda_used=False)


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1),
                         nn.GroupNorm(4, outgoing), nn.SiLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1),
                         nn.GroupNorm(4, outgoing), nn.SiLU())


class NativeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = Block(28, 16), Block(16, 32), Block(32, 64)
        self.d, self.e = Block(96, 32), Block(48, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        d = self.d(torch.cat((F.interpolate(c, size=b.shape[-2:], mode='bilinear',
                                           align_corners=False), b), 1))
        e = self.e(torch.cat((F.interpolate(d, size=a.shape[-2:], mode='bilinear',
                                           align_corners=False), a), 1))
        # Full 256 context, only central 128 is scored.
        return self.out(e)[:, 0, 64:192, 64:192]


def masked_core_loss(logits, target, mask):
    count = mask.flatten(1).sum(1)
    if torch.any(count <= 0):
        raise ValueError('Empty training mask')
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return ((loss * mask).flatten(1).sum(1) / count).mean()


def assemble_context(chunks, x, y):
    x0, y0 = x - 64, y - 64
    result = np.empty((28, 256, 256), dtype=np.uint8)
    for cy in range(y0 // 128, (y0 + 255) // 128 + 1):
        for cx in range(x0 // 128, (x0 + 255) // 128 + 1):
            a = chunks[(cy, cx)]
            left, top = max(x0, cx * 128), max(y0, cy * 128)
            right, bottom = min(x0 + 256, (cx + 1) * 128), min(y0 + 256, (cy + 1) * 128)
            result[:, top-y0:bottom-y0, left-x0:right-x0] = a[
                :, top-cy*128:bottom-cy*128, left-cx*128:right-cx*128]
    return result


def source_manifest(source):
    paths = set()
    for name in ['RESULTS.json', 'MASK_COVERAGE.json', 'MASK_STAGE_COMPLETE.json',
                 'REQUEST_LEDGER.json', 'SOURCE_PLAN.json', 'GEOMETRY_FOLD_CLASS_COUNTS.json',
                 'geometry/RESULTS.json', 'ct/REQUEST_PLAN.json', 'ct/support/RESULTS.json']:
        paths.add(source / name)
    for r in read(source / 'REQUEST_LEDGER.json')['requests']:
        p = source / r['path']
        assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256']
        paths.update([p, p.parents[2] / 'zarr.json'])
    for r in read(source / 'ct/REQUEST_PLAN.json')['requests']:
        rp = source / 'ct/receipts' / (r['id'] + '.json')
        receipt = read(rp)
        p = source / 'ct' / r['relative_path']
        assert receipt['bytes'] == r['expected_bytes'] == 28 * 128 * 128
        assert p.stat().st_size == receipt['bytes'] and sha(p) == receipt['sha256']
        paths.update([rp, p])
    for key, expected in read(source / 'GEOMETRY_FOLD_CLASS_COUNTS.json')['inputs_sha256'].items():
        assert sha(source / key) == expected
    return [pin(p, source) for p in sorted(paths)]


def load_data(source):
    sites = read(source / 'RESULTS.json')['sites']
    rows, xs, ys, masks = [], [], [], []
    for site in sites:
        w = site['winding']
        supervision = zarr.open_array(str(source / 'arrays' / w / 'supervision'), mode='r')
        ink = zarr.open_array(str(source / 'arrays' / w / 'inklabels'), mode='r')
        chunks = {}
        for r in read(source / 'ct/REQUEST_PLAN.json')['requests']:
            if r['winding'] == w:
                chunks[tuple(r['chunk_zyx'][1:])] = np.fromfile(
                    source / 'ct' / r['relative_path'], dtype=np.uint8).reshape(28, 128, 128)
        for r in site['rows']:
            if not r['eligible']:
                continue
            x, y = r['x'], r['y']
            context = assemble_context(chunks, x, y)
            valid = context.max(0) > 0
            assert valid.all(), 'Frozen context support changed'
            m = np.asarray(supervision[y:y+128, x:x+128]) > 0
            label = np.asarray(ink[y:y+128, x:x+128]) > 0
            assert int(m.sum()) == r['supervised_pixels']
            assert int((m & label).sum()) == r['ink_supervised']
            rows.append(dict(r, winding=w))
            xs.append(context)
            ys.append(label)
            masks.append(m)
    assert len(rows) == 265 and sum(int(a.sum()) for a in masks) == 2825834
    return dict(rows=rows, x=np.stack(xs), y=np.stack(ys), mask=np.stack(masks))


def fold_indices(data, fold):
    lookup = {r['core_id']: i for i, r in enumerate(data['rows'])}
    train = [lookup[k] for k in fold['retained_training_core_ids']]
    test = [lookup[k] for k in fold['original_evaluation_core_ids']]
    assert set(train).isdisjoint(test)
    assert all(data['rows'][i]['winding'] != fold['held_out'] for i in train)
    assert all(data['rows'][i]['winding'] == fold['held_out'] for i in test)
    assert fold['fold_geometry_admitted']
    return train, test


def normalization(data, indices):
    # Constant population moments across TRAINING context occurrences only.
    total = np.zeros(28, np.float64)
    squares = np.zeros(28, np.float64)
    for i in indices:
        a = data['x'][i].astype(np.float64) / 255.0
        total += a.sum((1, 2))
        squares += (a * a).sum((1, 2))
    n = len(indices) * 256**2
    mean = total / n
    std = np.maximum(np.sqrt(np.maximum(squares / n - mean**2, 0)), 1 / 255)
    return dict(mean=mean.tolist(), std=std.tolist(), context_occurrences=len(indices), pixels=n)


def normalized(x, norm):
    a = torch.as_tensor(x.astype(np.float32)) / 255.0
    return (a - torch.tensor(norm['mean'], dtype=torch.float32)[None, :, None, None]) / torch.tensor(
        norm['std'], dtype=torch.float32)[None, :, None, None]


def make_batch(data, indices, norm, rng, batch_size):
    counts = np.array([int(data['mask'][i].sum()) for i in indices], np.float64)
    picked = rng.choice(indices, size=batch_size, replace=True, p=counts/counts.sum())
    x = normalized(data['x'][picked], norm)
    y = torch.as_tensor(data['y'][picked].astype(np.float32))
    m = torch.as_tensor(data['mask'][picked].astype(np.float32))
    for b in range(batch_size):
        k, flip = int(rng.integers(4)), int(rng.integers(2))
        for a in (x, y, m):
            v = torch.rot90(a[b], k, dims=(-2, -1))
            if flip:
                v = torch.flip(v, dims=(-1,))
            a[b] = v.clone()
    return x, y, m, [int(i) for i in picked]


def train_step(model, optimizer, data, indices, norm, rng, batch_size):
    model.train()
    x, y, m, picked = make_batch(data, indices, norm, rng, batch_size)
    optimizer.zero_grad(set_to_none=True)
    loss = masked_core_loss(model(x), y, m)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), picked, int(m.sum())


def ranking_metrics(y, score):
    y = np.asarray(y, dtype=bool).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    assert y.size == s.size and y.size and np.isfinite(s).all()
    p, n = int(y.sum()), int((~y).sum())
    if p == 0 or n == 0:
        return dict(ap=None, auc=None, positives=p, negatives=n, undefined_single_class=True)
    order = np.argsort(-s, kind='stable')
    end = np.r_[np.flatnonzero(s[order][1:] != s[order][:-1]), y.size-1]
    tp = np.cumsum(y[order], dtype=np.int64)[end]
    fp = end+1-tp
    ap = np.sum(np.diff(np.r_[0, tp])/p * (tp/(tp+fp)))
    auc = np.trapezoid(np.r_[0, tp/p], np.r_[0, fp/n])
    return dict(ap=float(ap), auc=float(auc), positives=p, negatives=n,
                undefined_single_class=False)


def metrics(y, logits):
    result = ranking_metrics(y, logits)
    logits = np.asarray(logits, np.float64)
    y = np.asarray(y, np.float64)
    result.update(bce=float(np.mean(np.logaddexp(0, logits)-y*logits)),
                  prevalence=float(y.mean()), mean_probability=float(expit(logits).mean()),
                  fraction_probability_ge_half=float((logits >= 0).mean()))
    return result


@torch.inference_mode()
def predict(model, data, indices, norm, batch_size=4):
    model.eval()
    arrays = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start+batch_size]
        arrays.append(model(normalized(data['x'][ids], norm)).numpy().copy())
    return np.concatenate(arrays)


def score_maps(data, indices, maps):
    labels, scores, rows = [], [], []
    for j, i in enumerate(indices):
        mask = data['mask'][i]
        y, s = data['y'][i][mask], maps[j][mask]
        labels.append(y)
        scores.append(s)
        rows.append(dict(core_id=data['rows'][i]['core_id'], **metrics(y, s)))
    return dict(overall=metrics(np.concatenate(labels), np.concatenate(scores)), cores=rows)


def full_train_loss(model, data, indices, norm):
    maps = predict(model, data, indices, norm)
    y = np.concatenate([data['y'][i][data['mask'][i]] for i in indices])
    s = np.concatenate([maps[j][data['mask'][i]] for j, i in enumerate(indices)])
    return metrics(y, s)


def profile_features(data, indices, norm):
    xs, ys = [], []
    for i in indices:
        a = data['x'][i, :, 64:192, 64:192].transpose(1, 2, 0)[data['mask'][i]]
        xs.append((a.astype(np.float64)/255 - np.asarray(norm['mean'])) / np.asarray(norm['std']))
        ys.append(data['y'][i][data['mask'][i]].astype(np.float64))
    return np.concatenate(xs), np.concatenate(ys)


def fit_profile(data, indices, norm, plan):
    x, y = profile_features(data, indices, norm)
    penalty = plan['profile_l2']
    trace = []
    def objective(beta):
        z = x @ beta[:28] + beta[28]
        residual = expit(z)-y
        value = np.mean(np.logaddexp(0, z)-y*z) + penalty/2 * np.dot(beta[:28], beta[:28])
        gradient = np.r_[x.T @ residual / y.size + penalty*beta[:28], residual.mean()]
        return float(value), gradient
    initial = np.zeros(29)
    initial[28] = math.log(y.mean()/(1-y.mean()))
    def callback(beta):
        value, gradient = objective(beta)
        trace.append(dict(objective=value, gradient_infinity=float(abs(gradient).max())))
    result = minimize(objective, initial, method='L-BFGS-B', jac=True, callback=callback,
                      options=dict(maxiter=plan['profile_max_iterations'], gtol=1e-7, ftol=1e-12,
                                   maxls=40))
    value, gradient = objective(result.x)
    return dict(beta=result.x.tolist(), prevalence=float(y.mean()), train=metrics(y, x @ result.x[:28]+result.x[28]),
                iterations=int(result.nit), scipy_success=bool(result.success), message=str(result.message),
                gradient_infinity=float(abs(gradient).max()), converged=bool(abs(gradient).max() <= 1e-5),
                regularized_objective=value, trace=trace)


def update_plateau(state, loss, plan):
    if state['significant_best'] - loss >= plan['plateau_min_delta']:
        state['significant_best'], state['bad_checks'] = loss, 0
    else:
        state['bad_checks'] += 1
    if state['lr'] > plan['minimum_lr'] and state['bad_checks'] >= plan['lr_patience_checks']:
        state['lr'] = max(plan['minimum_lr'], state['lr']*plan['lr_factor'])
        state['bad_checks'] = 0
    return (state['step'] >= plan['minimum_updates'] and state['lr'] == plan['minimum_lr']
            and state['bad_checks'] >= plan['stop_patience_checks'])


def initial_state(seed, initial_metrics, plan):
    return dict(step=0, seed=seed, lr=plan['learning_rate'], bad_checks=0,
                significant_best=initial_metrics['bce'], best_loss=initial_metrics['bce'], best_step=0,
                draws=0, supervised_pixel_exposures=0, unique_core_indices=[],
                history=[dict(step=0, lr=plan['learning_rate'], **initial_metrics)],
                termination=None)


def commit_checkpoint(directory, binding, model, optimizer, rng, state):
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['step']:06d}"
    file = directory / (name + '.pt')
    receipt = directory / (name + '.json')
    if receipt.exists():
        raise RuntimeError('Refusing to overwrite committed checkpoint')
    tmp = file.with_suffix('.pt.partial')
    payload = dict(binding=binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   numpy_rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                   state=copy.deepcopy(state))
    with tmp.open('wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, file)
    row = dict(binding=binding, checkpoint=pin(file, directory), step=state['step'],
               best_step=state['best_step'], train=state['history'][-1],
               termination=state['termination'], committed_utc=now())
    atomic_json(receipt, row)  # Publication marker is always last.
    return row


def load_checkpoint(directory, receipt, binding, model, optimizer=None):
    row = read(receipt)
    assert row['binding'] == binding
    path = verified(directory, row['checkpoint'])
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['binding'] == binding and payload['state']['step'] == row['step']
    model.load_state_dict(payload['model'])
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = payload['numpy_rng']
    torch.set_rng_state(payload['torch_rng'])
    return payload['state'], rng


def validate_go(run, source):
    go = read(run / 'GO.json')
    for row in go['code_pins']:
        verified(run, row)
    if environment() != go['environment']:
        raise RuntimeError('Runtime differs from frozen environment; qualify amendment before resume')
    for row in go['source_pins']:
        verified(source, row)
    return go


def freeze(run, source):
    assert not (run / 'GO.json').exists()
    qualification = read(run / 'QUALIFICATION.json')
    assert qualification['all_passed'] and qualification['environment'] == environment()
    plan = read(run / 'PLAN.json')
    pins = source_manifest(source)
    atomic_json(run / 'GO.json', dict(created_utc=now(), plan=plan, environment=environment(),
                source_archives=read(run/'INPUT_ARCHIVES.json'),
                source_pins=pins, code_pins=[pin(run / n, run) for n in
                    ['native_learning.py', 'qualify.py', 'backup.py', 'restore_backup.py', 'status.py',
                     'PROTOCOL.md', 'PLAN.json', 'QUALIFICATION.json', 'RUNBOOK.md', 'requirements.txt',
                     'INPUT_ARCHIVES.json', 'DATA_QUALIFICATION.json']],
                training_started_before_freeze=False, held_out_predictions_before_freeze=0))


def run_arm(run, data, train, norm, fold, seed, plan, go_sha):
    name = f'{fold}-seed{seed}'
    directory = run / 'work' / name
    binding = dict(go_sha256=go_sha, fold=fold, seed=seed,
                   normalization_sha256=sha(run/'work'/fold/'NORMALIZATION.json'))
    torch.manual_seed(seed)
    model = NativeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'])
    receipts = sorted(directory.glob('step-*.json'))
    if receipts:
        state, rng = load_checkpoint(directory, receipts[-1], binding, model, optimizer)
    else:
        rng = np.random.default_rng(seed)
        state = initial_state(seed, full_train_loss(model, data, train, norm), plan)
        commit_checkpoint(directory, binding, model, optimizer, rng, state)
    if (directory/'COMPLETE.json').exists():
        completion = read(directory/'COMPLETE.json')
        assert completion['binding'] == binding and state['termination']
        verified(directory, completion['endpoint_receipt'])
        return
    unique = set(state['unique_core_indices'])
    started = time.monotonic()
    initial_step = state['step']
    while not state['termination']:
        loss, picked, pixels = train_step(model, optimizer, data, train, norm, rng, plan['batch_size'])
        state['step'] += 1
        state['draws'] += len(picked)
        state['supervised_pixel_exposures'] += pixels
        unique.update(picked)
        if state['step'] % plan['check_every_updates']:
            continue
        result = full_train_loss(model, data, train, norm)
        state['history'].append(dict(step=state['step'], lr=state['lr'], **result))
        if result['bce'] < state['best_loss']:
            state['best_loss'], state['best_step'] = result['bce'], state['step']
        stop = update_plateau(state, result['bce'], plan)
        for group in optimizer.param_groups:
            group['lr'] = state['lr']
        if stop:
            state['termination'] = 'training_loss_plateau_at_minimum_lr'
        elif state['step'] >= plan['maximum_updates']:
            old = [r['bce'] for r in state['history'] if r['step'] <= state['step']-1024]
            gain = min(old)-state['best_loss']
            state['termination'] = 'ceiling_still_improving' if gain >= plan['plateau_min_delta'] else 'ceiling_without_confirmed_plateau'
        state['unique_core_indices'] = sorted(unique)
        commit_checkpoint(directory, binding, model, optimizer, rng, state)
        status = dict(utc=now(), phase='training', arm=name, step=state['step'],
                      maximum_updates=plan['maximum_updates'], train=result, lr=state['lr'],
                      seconds_per_update_including_training_checks=(time.monotonic()-started)/(state['step']-initial_step),
                      termination=state['termination'], held_out_predictions=0,
                      draws=state['draws'], unique_cores=len(unique),
                      supervised_pixel_exposures=state['supervised_pixel_exposures'])
        atomic_json(run/'STATUS.json', status)
        print(json.dumps(status), flush=True)
    endpoint = directory / f"step-{state['best_step']:06d}.json"
    atomic_json(directory/'COMPLETE.json', dict(binding=binding, terminal_step=state['step'],
                termination=state['termination'], endpoint_step=state['best_step'],
                endpoint_receipt=pin(endpoint, directory), history=state['history'],
                draws=state['draws'], supervised_pixel_exposures=state['supervised_pixel_exposures'],
                completed_utc=now()))


def profile_maps(data, indices, norm, beta):
    result = []
    b = np.array(beta)
    for i in indices:
        a = data['x'][i, :, 64:192, 64:192].astype(np.float64)/255
        a = (a-np.array(norm['mean'])[:, None, None])/np.array(norm['std'])[:, None, None]
        result.append(np.einsum('c,cyx->yx', b[:28], a)+b[28])
    return np.stack(result)


def save_maps(path, maps):
    tmp = path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:
        np.save(f, maps, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def evaluate(run, source, data, folds, plan, go_sha):
    # No prediction allowed until every predeclared training endpoint is committed.
    required = [run/'work'/f"{f['held_out']}-seed{s}"/'COMPLETE.json'
                for f in folds for s in plan['seeds']]
    assert all(p.is_file() for p in required)
    frozen_endpoints = [pin(p, run) for p in required]
    eval_dir = run/'evaluation'
    eval_dir.mkdir(exist_ok=True)
    seal = dict(go_sha256=go_sha, endpoints=frozen_endpoints)
    if (eval_dir/'ENDPOINTS.json').exists():
        assert read(eval_dir/'ENDPOINTS.json') == seal
    else:
        atomic_json(eval_dir/'ENDPOINTS.json', seal)
    outputs = []
    for fold in folds:
        w = fold['held_out']
        train, test = fold_indices(data, fold)
        norm = read(run/'work'/w/'NORMALIZATION.json')['normalization']
        base = read(run/'work'/w/'PROFILE.json')
        for name in ['constant', 'profile'] + [f'cnn-seed{s}' for s in plan['seeds']]:
            receipt = eval_dir/f'{w}-{name}.json'
            if receipt.exists():
                row = read(receipt)
                assert row['go_sha256'] == go_sha
                verified(eval_dir, row['maps'])
                outputs.append(row)
                continue
            if name == 'constant':
                p = base['fit']['prevalence']
                maps = np.full((len(test),128,128), math.log(p/(1-p)), np.float64)
            elif name == 'profile':
                maps = profile_maps(data, test, norm, base['fit']['beta'])
            else:
                seed = int(name.split('seed')[1])
                arm = run/'work'/f'{w}-seed{seed}'
                completion = read(arm/'COMPLETE.json')
                endpoint = verified(arm, completion['endpoint_receipt'])
                model = NativeNet()
                load_checkpoint(arm, endpoint, completion['binding'], model)
                maps = predict(model, data, test, norm)
            file = eval_dir/f'{w}-{name}.npy'
            save_maps(file, maps)
            row = dict(go_sha256=go_sha, fold=w, model=name, core_ids=[data['rows'][i]['core_id'] for i in test],
                       maps=pin(file, eval_dir), **score_maps(data, test, maps))
            atomic_json(receipt, row)
            outputs.append(row)
    summary = []
    for fold in folds:
        w = fold['held_out']
        selected = [r for r in outputs if r['fold'] == w]
        profile = next(r['overall'] for r in selected if r['model'] == 'profile')
        cnn_rows = [r for r in selected if r['model'].startswith('cnn-')]
        cnns = [r['overall'] for r in cnn_rows]
        def blank_mean(row):
            blank = [r for r in row['cores'] if r['positives'] == 0]
            return sum(r['negatives']*r['mean_probability'] for r in blank)/sum(r['negatives'] for r in blank)
        profile_blank = blank_mean(next(r for r in selected if r['model'] == 'profile'))
        fitting = []
        for seed in plan['seeds']:
            complete = read(run/'work'/f'{w}-seed{seed}'/'COMPLETE.json')
            endpoint = next(r for r in complete['history'] if r['step'] == complete['endpoint_step'])
            prevalence = endpoint['prevalence']
            constant_bce = -prevalence*math.log(prevalence)-(1-prevalence)*math.log(1-prevalence)
            reduction = 1-endpoint['bce']/constant_bce
            fitting.append(dict(seed=seed, endpoint_step=complete['endpoint_step'],
                                terminal_step=complete['terminal_step'], termination=complete['termination'],
                                train_ap=endpoint['ap'], train_bce_reduction_from_constant=reduction,
                                adequate_fit=bool(endpoint['ap'] >= plan['adequate_training_ap'] and
                                reduction >= plan['adequate_training_bce_reduction_from_constant'])))
        summary.append(dict(fold=w, profile=profile, cnn_mean_ap=float(np.mean([r['ap'] for r in cnns])),
                            cnn_ap_min=float(min(r['ap'] for r in cnns)), cnn_ap_max=float(max(r['ap'] for r in cnns)),
                            mean_ap_gain=float(np.mean([r['ap'] for r in cnns])-profile['ap']),
                            all_seeds_gain=all(r['ap'] > profile['ap'] for r in cnns), training=fitting,
                            profile_blank_mean_probability=profile_blank,
                            cnn_blank_mean_probabilities=[blank_mean(r) for r in cnn_rows],
                            blank_response_gate=all(blank_mean(r)-profile_blank <= plan['blank_core_max_probability_increase'] for r in cnn_rows)))
    transfer_gate = all(r['all_seeds_gain'] and r['mean_ap_gain'] >= plan['advance_ap_margin'] for r in summary)
    fit_gate = all(f['adequate_fit'] for r in summary for f in r['training'])
    blank_gate = all(r['blank_response_gate'] for r in summary)
    budget_limited = any(f['termination'] != 'training_loss_plateau_at_minimum_lr' for r in summary for f in r['training'])
    if transfer_gate and fit_gate and blank_gate:
        decision = 'advance_to_broader_paired_supervision_design'
    elif budget_limited:
        decision = 'optimization_not_demonstrated_saturated_no_native_signal_null'
    elif fit_gate and not transfer_gate:
        decision = 'fit_without_consistent_transfer_prioritize_coverage_or_reference_mismatch'
    elif not fit_gate:
        decision = 'fit_inadequate_one_training_only_representation_optimizer_diagnosis'
    else:
        decision = 'blank_response_guard_failed_review_calibration_and_transfer'
    result = dict(go_sha256=go_sha, completed_utc=now(), folds=summary,
                  descriptive_macro_ap_gain=float(np.mean([r['mean_ap_gain'] for r in summary])),
                  practical_transfer_gate=bool(transfer_gate), adequate_fit_gate=fit_gate,
                  blank_response_gate=blank_gate, decision=decision,
                  decision_scope='Prospective practical priorities, not a significance test, ink truth or GPU authorization',
                  outputs=[pin(eval_dir/f"{r['fold']}-{r['model']}.json", run) for r in outputs])
    atomic_json(run/'RESULTS.json', result)
    atomic_json(run/'STATUS.json', dict(utc=now(), phase='complete', results_sha256=sha(run/'RESULTS.json')))


def execute(run, source):
    lock = (run/'WORKER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(run/'WORKER.json', dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
                started_utc=now(), resume_command=f'python native_learning.py run --run {run} --source {source}'))
    go = validate_go(run, source)
    go_sha = sha(run/'GO.json')
    plan = go['plan']
    atomic_json(run/'STATUS.json', dict(utc=now(), phase='loading_verified_inputs', held_out_predictions=0))
    data = load_data(source)
    folds = read(source/'geometry/RESULTS.json')['folds']
    assert [f['held_out'] for f in folds] == plan['fold_order']
    # Fit every baseline first, but do not calculate held-out baseline metrics.
    for fold in folds:
        w = fold['held_out']
        train, test = fold_indices(data, fold)
        directory = run/'work'/w
        directory.mkdir(parents=True, exist_ok=True)
        norm_path = directory/'NORMALIZATION.json'
        if norm_path.exists():
            frozen = read(norm_path)
            assert frozen['go_sha256'] == go_sha and frozen['training_core_ids'] == fold['retained_training_core_ids']
            norm = frozen['normalization']
        else:
            norm = normalization(data, train)
            atomic_json(norm_path, dict(go_sha256=go_sha, training_core_ids=fold['retained_training_core_ids'], normalization=norm))
        if not (directory/'PROFILE.json').exists():
            atomic_json(run/'STATUS.json', dict(utc=now(), phase='fitting_profile', fold=w, held_out_predictions=0))
            fit = fit_profile(data, train, norm, plan)
            atomic_json(directory/'PROFILE.json', dict(go_sha256=go_sha, normalization_sha256=sha(norm_path), fit=fit))
        base = read(directory/'PROFILE.json')
        assert base['go_sha256'] == go_sha and base['normalization_sha256'] == sha(norm_path)
        if not base['fit']['converged']:
            raise RuntimeError('Profile baseline not converged; preserve failed study for diagnosis')
    for seed in plan['seeds']:
        for fold in folds:
            w = fold['held_out']
            train, _ = fold_indices(data, fold)
            norm = read(run/'work'/w/'NORMALIZATION.json')['normalization']
            atomic_json(run/'STATUS.json', dict(utc=now(), phase='starting_or_resuming_arm', arm=f'{w}-seed{seed}', held_out_predictions=0))
            run_arm(run, data, train, norm, w, seed, plan, go_sha)
    evaluate(run, source, data, folds, plan, go_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    setup()
    args.run.mkdir(exist_ok=True, parents=True)
    if args.stage == 'freeze':
        freeze(args.run, args.source)
    else:
        try:
            execute(args.run, args.source)
        except BaseException as e:
            atomic_json(args.run/'ERROR.json', dict(utc=now(), exception=type(e).__name__, message=str(e)))
            raise


if __name__ == '__main__':
    main()
