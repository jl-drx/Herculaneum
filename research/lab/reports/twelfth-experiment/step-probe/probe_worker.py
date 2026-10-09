"""Three independent first AdamW steps from an identical frozen initializer/batch."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import socket
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def offline(*args, **kwargs):
    raise RuntimeError('Network forbidden in matched first-step probe')


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest}


def verify(pin):
    actual = fingerprint(pin['path'])
    require(all(actual[k] == pin[k] for k in actual), 'Pinned input changed:' + Path(pin['path']).name)
    return Path(pin['path'])


def module(pin, name):
    path = verify(pin)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def mode_guard(model, metadata, torch):
    require(metadata['embedded_config']['targets']['ink']['activation'] == 'none', 'Output activation changed')
    require(not list(model.named_buffers()), 'Mutable buffers not supported')
    for _, layer in model.named_modules():
        require(not isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
                and not getattr(layer, 'track_running_stats', False), 'Running normalization not supported')
        if isinstance(layer, torch.nn.modules.dropout._DropoutNd):
            require(layer.p == 0, 'Stochastic dropout not supported')
        if type(layer).__name__ in ['DropPath', 'StochasticDepth']:
            require(getattr(layer, 'drop_prob', getattr(layer, 'p', 0)) == 0, 'Stochastic depth not supported')


def model_logits(model, tensor, torch):
    value = model(tensor)['ink']
    logits = value[0] if isinstance(value, (tuple, list)) else value
    require(logits.dtype == torch.float32 and tuple(logits.shape) == (2, 1, 128, 128)
            and bool(torch.isfinite(logits).all()), 'Output logits contract changed')
    return logits


def batch(plan, shared, mechanics, runtime, torch, np):
    schedule = json.loads(verify(plan['schedule']).read_text())
    require(len(schedule) == 32 and schedule[0]['step'] == 1, 'Historical schedule changed')
    step = schedule[0]
    require(step['patches'] == plan['historical']['A']['smoothed_first_step']['patches']
            == plan['historical']['B']['smoothed_first_step']['patches'], 'Historical first patches changed')
    regions = [shared.training_region(c) for c in plan['adaptation']]
    images, masks, targets, exposures = [], [[], []], [[], []], []
    for patch in step['patches']:
        r = patch['region']
        data, manual, hard_target, pseudo, pseudo_target = regions[r]
        x, y, width, height = patch['bbox']
        raw = data['surface'][2:19, y:y + height, x:x + width]
        image = np.ascontiguousarray(runtime['normalize_robust'](raw), dtype=np.float32)
        require(np.isfinite(image).all(), 'Normalizer produced nonfinite input')
        images.append(mechanics.augment(image, patch['augmentation'])[None])
        exposure = {'region': r}
        for i, (mask, target, name) in enumerate([(manual, hard_target, 'manual'),
                                                (pseudo, pseudo_target, 'pseudo')]):
            mm = mechanics.augment(mask[y:y + height, x:x + width], patch['augmentation'])
            tt = mechanics.augment(target[y:y + height, x:x + width], patch['augmentation'])
            masks[i].append(mm[None])
            targets[i].append(tt[None])
            exposure[name] = {'positive': int((mm & (tt == 1)).sum()),
                              'negative': int((mm & (tt == 0)).sum())}
        exposures.append(exposure)
    tensor = torch.from_numpy(np.stack(images))
    m, p = [torch.from_numpy(np.stack(v)) for v in masks]
    t, pt = [torch.from_numpy(np.stack(v)) for v in targets]
    identity = shared.state_hash({'input': tensor, 'manual_mask': m, 'manual_hard_target': t,
                                 'pseudo_mask': p, 'pseudo_hard_target': pt})
    require(identity == '571d10c58008cb6f9e581a717bbb067846dae6a2ee2d29bdf9eb141ca9a02ec3',
            'Exact historical normalized input/mask/target fingerprint changed')
    return tensor, m, t, p, pt, step, exposures, identity


def losses_record(z, initial, m, t, p, pt, objective, smoothing, preservation, mechanics, torch):
    u, pseudo, balanced = smoothing.smoothed_losses(z, m, t, p, pt, .5)
    hard_u, hard_p, _ = mechanics.losses(z, m, t, p, pt)
    dm, dp = preservation.preservation_losses(z, initial, m, p)
    coefficient = 0. if objective == 'A' else .25
    base = u + coefficient * pseudo
    divergence = dm + coefficient * dp
    return {'smooth_U_fp32': float(u), 'smooth_P_fp32': float(pseudo), 'smooth_R_fp32': float(balanced),
            'base_fp32': float(base), 'kl_M_fp64': float(dm), 'kl_P_fp64': float(dp),
            'weighted_kl_fp64': float(divergence), 'full_mixed_dtype': float(base.double() + divergence),
            'hard_U_fp32': float(hard_u), 'hard_P_fp32': float(hard_p),
            'hard_base_fp32': float(hard_u + coefficient * hard_p)}


def stable_changes(z, initial, m, t, p, pt, objective, torch):
    z, initial = z.double(), initial.double()
    delta = z - initial
    q = torch.sigmoid(initial)
    def term(mask, target):
        za, ta, qa, dz = z[mask], initial[mask], q[mask], delta[mask]
        s = .25 + .5 * target[mask].double()
        k = (torch.logaddexp(za, torch.zeros_like(za))
             - torch.logaddexp(ta, torch.zeros_like(ta)) - qa * dz).mean()
        linear = ((qa - s) * dz).mean()
        direct = ((torch.logaddexp(za, torch.zeros_like(za)) - s * za)
                  - (torch.logaddexp(ta, torch.zeros_like(ta)) - s * ta)).mean()
        hard = ((torch.logaddexp(za, torch.zeros_like(za)) - target[mask].double() * za)
                - (torch.logaddexp(ta, torch.zeros_like(ta)) - target[mask].double() * ta)).mean()
        return {'kl': float(k), 'linear': float(linear), 'base_delta': float(direct),
                'hard_delta': float(hard), 'identity_error': abs(float(direct - k - linear))}
    manual, pseudo = term(m, t), term(p, pt)
    coefficient = 0. if objective == 'A' else .25
    weighted = {key: manual[key] + coefficient * pseudo[key]
                for key in ['kl', 'linear', 'base_delta', 'hard_delta']}
    weighted['full_delta'] = weighted['base_delta'] + weighted['kl']
    weighted['identity_error'] = abs(weighted['base_delta'] - weighted['kl'] - weighted['linear'])
    require(min(manual['kl'], pseudo['kl']) >= -1e-6, 'Nonfinite/negative KL beyond roundoff')
    require(max(manual['identity_error'], pseudo['identity_error'], weighted['identity_error']) <= 1e-10,
            'Stable base-change decomposition failed')
    return {'manual': manual, 'pseudo': pseudo, 'weighted': weighted,
            'identity': 'delta_base=KL+linear;delta_full=2KL+linear',
            'roundoff_policy': 'No clipping of tiny KL/loss changes; retain raw signs/magnitudes'}


def parameter_stats(model, initial, learning_rate, optimizer, torch):
    records = []
    dot, ideal_norm2, actual_norm2, max_error, max_ratio = 0., 0., 0., 0., 0.
    changed, unchanged, wrong_sign = 0, 0, 0
    for name, parameter in model.named_parameters():
        before = initial[name]
        actual = parameter.detach().double() - before.double()
        if parameter.grad is None:
            require(torch.equal(parameter.detach(), before), 'Parameter without gradient changed')
            records.append({'name': name, 'gradient': None, 'unchanged': True})
            continue
        g = parameter.grad.detach().double()
        ideal = -learning_rate * g / (g.abs() + 1e-8)
        ulp = (torch.nextafter(before, torch.full_like(before, float('inf'))) - before).abs().double()
        allowance = 4 * ulp + 8 * torch.finfo(torch.float32).eps * ideal.abs() + 1e-12
        error = (actual - ideal).abs()
        ratio = error / allowance
        require(bool(torch.isfinite(parameter).all()) and bool((ratio <= 1).all()),
                'Analytic first AdamW update exceeds the predeclared FP32 allowance')
        state = optimizer.state[parameter]
        require(float(state['step']) == 1, 'Optimizer moments were not fresh')
        wrong = int(((g * actual) > 0).sum())
        require(wrong == 0, 'A parameter moved against the clipped gradient descent direction')
        gd = float((g * actual).sum())
        dot += gd
        ideal_norm2 += float(ideal.square().sum())
        actual_norm2 += float(actual.square().sum())
        local_changed = int((actual != 0).sum())
        changed += local_changed
        unchanged += int(actual.numel()) - local_changed
        wrong_sign += wrong
        local_error, local_ratio = float(error.max()), float(ratio.max())
        max_error, max_ratio = max(max_error, local_error), max(max_ratio, local_ratio)
        index = int(g.abs().reshape(-1).argmax())
        records.append({'name': name, 'shape': list(parameter.shape), 'numel': parameter.numel(),
                        'clipped_gradient_l2': float(g.square().sum().sqrt()),
                        'ideal_update_l2': float(ideal.square().sum().sqrt()),
                        'actual_update_l2': float(actual.square().sum().sqrt()),
                        'gradient_dot_actual_update': gd, 'changed_coordinates': local_changed,
                        'max_absolute_identity_error': local_error, 'max_error_allowance_ratio': local_ratio,
                        'optimizer_step': float(state['step']),
                        'probe': {'index': index, 'before': float(before.reshape(-1)[index]),
                                  'after': float(parameter.detach().reshape(-1)[index]),
                                  'clipped_gradient': float(g.reshape(-1)[index]),
                                  'ideal_delta': float(ideal.reshape(-1)[index])}})
    require(dot <= 0, 'Positive aggregate gradient-dot-update is an error')
    return {'per_parameter': records, 'gradient_dot_actual_update': dot,
            'ideal_update_l2': ideal_norm2**.5, 'actual_update_l2': actual_norm2**.5,
            'changed_coordinates': changed, 'unchanged_coordinates': unchanged,
            'wrong_sign_coordinates': wrong_sign, 'max_absolute_identity_error': max_error,
            'max_error_allowance_ratio': max_ratio,
            'identity': '-lr*g_clipped/(abs(g_clipped)+1e-8)',
            'allowance': '4*ULP(initial_fp32)+8*eps_fp32*abs(ideal_delta)+1e-12',
            'roundoff_note': 'FP32 unchanged coordinates can round to zero update; no relative-error test'}


def execute(args):
    # Guards precede NumPy/Torch/upstream imports and any binary input verification.
    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_CPU, (360, 360))
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    started = time.perf_counter()
    plan_pin = fingerprint(args.plan)
    require(plan_pin['sha256'] == args.plan_sha256, 'Frozen step-probe plan SHA mismatch')
    plan = json.loads(args.plan.read_text())
    require(plan['kind'] == 'matched-single-step-descent-v1' and plan['ready_for_execution'] is True,
            'Probe remains conditional/unfrozen')
    require(plan['learning_rates'] == [1e-6, 1e-5, 1e-4]
            and plan['objectives'] == {'A': {'manual': 1., 'pseudo': 0.}, 'B': {'manual': 1., 'pseudo': .25}},
            'Objective/rate scope changed')
    require(plan['training'] == {'optimizer': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8,
                               'weight_decay': 0., 'scheduler': None, 'clip_norm': 1.,
                               'clip_denominator_epsilon': 1e-6, 'seed': 20261008,
                               'smoothing_epsilon': .5, 'steps_per_branch': 1,
                               'fresh_optimizer_per_branch': True, 'shared_fixed_clipped_gradient': True},
            'First-step optimizer contract changed')
    require(plan['limits'] == {'worker_wall_seconds': 120, 'total_wall_seconds': 300,
                              'cpu_seconds': 360, 'threads': 4, 'inter_op_threads': 1,
                              'rss_acceptance_bytes': 3 * 1024**3, 'address_space_bytes': 12 * 1024**3,
                              'output_bytes': 64 * 1024**2, 'retries': 0}, 'Fixed resource limits changed')
    require(plan['tolerances'] == {'historical_loss_absolute': 1e-7, 'historical_gradient_norm_absolute': 1e-6,
                                  'historical_logits': 'bitwise identical FP32', 'initial_kl_absolute': 1e-12,
                                  'kl_minimum': -1e-6,
                                  'adam_update': '4*ULP(initial_fp32)+8*eps_fp32*abs(ideal_delta)+1e-12',
                                  'loss_change_identity_absolute': 1e-10}, 'Fixed numerical tolerances changed')
    require(plan['kl'] == {'orientation': 'initial_teacher||post_student', 'coefficient': 1.,
                          'temperature': 1., 'dtype': 'float64',
                          'support': 'A:meanM; B:meanM+.25meanP; no unknown pixels'}, 'KL contract changed')
    require(plan['runtime'] == {'python': '3.12.14', 'numpy': '2.5.3', 'torch': '2.12.0+cpu'}
            and sys.version_info[:3] == (3, 12, 14), 'Pinned Python/runtime contract changed')
    for pin in plan['verified_inputs']:
        verify(pin)
    require(verify(plan['code']['probe_worker.py']) == Path(__file__).resolve(), 'Executed worker source changed')
    base = json.loads(verify(plan['base_plan']).read_text())
    for key in ['initializer', 'helper', 'adaptation', 'schedule']:
        require(plan[key] == base[key], 'Historical matching field changed:' + key)
    require(plan['initial_state_sha256'] == base['historical_initial_state_sha256'], 'Initial state pin changed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    require(np.__version__ == plan['runtime']['numpy'] and torch.__version__ == plan['runtime']['torch'],
            'Pinned NumPy/Torch runtime changed')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261008)
    shared = module(plan['shared'], 'step_shared')
    mechanics = module(plan['mechanics'], 'training_mechanics')
    smoothing = module(plan['smoothing'], 'step_smoothing')
    preservation = module(plan['preservation'], 'step_preservation')
    helper = module(plan['helper'], 'step_strict_model_helper')
    runtime = helper.load_runtime()
    model, metadata = helper.build_model(Path(plan['initializer']['path']), plan['initializer'], runtime)
    mode_guard(model, metadata, torch)
    require(all(value.grad is None for value in model.parameters()), 'Initializer has preexisting gradients')
    model.train()
    require(shared.state_hash(model.state_dict()) == plan['initial_state_sha256'], 'Initializer semantic hash changed')
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    tensor, m, t, p, pt, step, exposures, input_hash = batch(plan, shared, mechanics, runtime, torch, np)
    for name, value in {'input': tensor, 'manual-mask': m, 'manual-hard-target': t,
                        'pseudo-mask': p, 'pseudo-hard-target': pt}.items():
        np.save(args.output_dir / (name + '.npy'), value.numpy(), allow_pickle=False)
    rng = torch.random.get_rng_state().clone()
    logits = model_logits(model, tensor, torch)
    require(torch.equal(rng, torch.random.get_rng_state()), 'Initial forward changed RNG')
    teacher = logits.detach().clone()
    np.save(args.output_dir / 'initial-logits.npy', teacher.numpy(), allow_pickle=False)
    historical = plan['historical'][args.objective]
    for pin in historical['initial_logits'].values():
        archived = np.load(verify(pin), mmap_mode='r', allow_pickle=False)
        require(archived.shape == (32, 2, 1, 128, 128) and archived.dtype == np.float32,
                'Archived initial logit record differs')
        require(np.array_equal(np.ascontiguousarray(archived[0]).view(np.uint32),
                               teacher.numpy().view(np.uint32)),
                'Archived first-batch logits are not bitwise identical')
    sentinel = json.loads(verify(historical['hard_first_step']).read_text())
    require(sentinel['input_target_sha256'] == input_hash, 'Archived input sentinel changed')
    with torch.no_grad():
        pre = losses_record(teacher, teacher, m, t, p, pt, args.objective,
                            smoothing, preservation, mechanics, torch)
    require(abs(pre['hard_U_fp32'] - sentinel['hard_U']) <= 1e-7
            and abs(pre['hard_P_fp32'] - sentinel['hard_P']) <= 1e-7, 'Historical hard BCE differs')
    old = historical['smoothed_first_step']
    require(abs(pre['smooth_U_fp32'] - old['loss_U']) <= 1e-7
            and abs(pre['smooth_P_fp32'] - old['loss_P']) <= 1e-7,
            'Historical smoothed BCE differs')
    require(max(abs(pre['kl_M_fp64']), abs(pre['kl_P_fp64'])) <= 1e-12, 'Initial KL is not zero')
    logits.retain_grad()
    u, pseudo, balanced = smoothing.smoothed_losses(logits, m, t, p, pt, .5)
    base_loss = u if args.objective == 'A' else u + .25 * pseudo
    require(all(value.grad is None for value in model.parameters()), 'Stale gradients before sole backward')
    base_loss.backward()
    active = m if args.objective == 'A' else m | p
    require(bool((logits.grad[~active] == 0).all()), 'Unknown output pixels received gradient')
    preclip_hash = shared.state_hash({n: value.grad for n, value in model.named_parameters() if value.grad is not None})
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    require(float(norm) > 0 and abs(float(norm) - old['preclip_global_gradient_norm']) <= 1e-6,
            'Historical initial gradient norm differs')
    gradient_hash = shared.state_hash({n: value.grad for n, value in model.named_parameters() if value.grad is not None})
    coefficient = float(torch.clamp(1. / (norm + 1e-6), max=1.))
    clipped_norm = sum(float(value.grad.detach().double().square().sum()) for value in model.parameters()
                       if value.grad is not None)**.5
    del logits, u, pseudo, balanced, base_loss
    gc.collect()
    trials = []
    for index, learning_rate in enumerate(plan['learning_rates']):
        directory = args.output_dir / f'lr-{index + 1}'
        directory.mkdir()
        with torch.no_grad():
            for name, value in model.named_parameters():
                value.copy_(initial[name])
        require(shared.state_hash(model.state_dict()) == plan['initial_state_sha256'], 'Reset state differs from initializer')
        require(shared.state_hash({n: value.grad for n, value in model.named_parameters() if value.grad is not None})
                == gradient_hash, 'Fixed initial clipped gradient changed before branch')
        torch.random.set_rng_state(rng)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(.9, .999),
                                     eps=1e-8, weight_decay=0)
        require(not optimizer.state, 'Optimizer carries historical moments')
        optimizer.step()
        updates = parameter_stats(model, initial, learning_rate, optimizer, torch)
        require(shared.state_hash({n: value.grad for n, value in model.named_parameters() if value.grad is not None})
                == gradient_hash, 'Optimizer mutated the fixed clipped gradient')
        with torch.no_grad():
            post = model_logits(model, tensor, torch).detach()
            require(torch.equal(rng, torch.random.get_rng_state()), 'Post-step forward changed RNG')
            measured = losses_record(post, teacher, m, t, p, pt, args.objective,
                                     smoothing, preservation, mechanics, torch)
            changes = stable_changes(post, teacher, m, t, p, pt, args.objective, torch)
        np.save(directory / 'post-logits.npy', post.numpy(), allow_pickle=False)
        write_json(directory / 'parameter-update-summary.json', updates)
        trial = {'learning_rate': learning_rate, 'initial_state_sha256': plan['initial_state_sha256'],
                 'post_state_sha256': shared.state_hash(model.state_dict()),
                 'clipped_gradient_sha256': gradient_hash, 'optimizer_initial_state_empty': True,
                 'optimizer_step': 1, 'pre': pre, 'post': measured, 'stable_change': changes,
                 'gradient_dot_actual_update': updates['gradient_dot_actual_update'],
                 'actual_update_l2': updates['actual_update_l2'],
                 'max_first_adam_error_allowance_ratio': updates['max_error_allowance_ratio'],
                 'post_logits': fingerprint(directory / 'post-logits.npy'),
                 'parameter_update_summary': fingerprint(directory / 'parameter-update-summary.json')}
        write_json(directory / 'result.json', trial)
        trials.append(trial)
        del optimizer, post
        gc.collect()
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 3 * 1024**3,
                'Peak RSS exceeded3GiB')
    for pin in plan['verified_inputs']:
        verify(pin)
    require(fingerprint(args.plan) == plan_pin, 'Frozen plan changed during probe')
    files = [fingerprint(path) for path in sorted(args.output_dir.rglob('*')) if path.is_file()]
    require(sum(x['size_bytes'] for x in files) < 64 * 1024**2, 'Worker output exceeded64MiB')
    result = {'schema_version': 1, 'status': 'PASS', 'objective': args.objective, 'plan': plan_pin,
              'initializer': plan['initializer'], 'initial_state_sha256': plan['initial_state_sha256'],
              'model_contract': metadata, 'first_step': step, 'exposures': exposures,
              'input_target_sha256': input_hash, 'initial_logits': fingerprint(args.output_dir / 'initial-logits.npy'),
              'preclip_gradient_sha256': preclip_hash, 'preclip_global_gradient_norm': float(norm),
              'actual_clip_coefficient': coefficient, 'clipped_global_gradient_norm_fp64': clipped_norm,
              'clipped_gradient_sha256': gradient_hash, 'trials': trials, 'outputs': files,
              'actual_calls': {'model_forwards': 4, 'backwards': 1, 'independent_fresh_optimizer_steps': 3},
              'checkpoint_outputs': 0, 'network_calls': 0, 'new_evaluation_or_calibration_calls': 0,
              'interpretation': 'Same-batch local first-step diagnostic; no winning LR or generalization claim',
              'torch_version': torch.__version__, 'wall_seconds': time.perf_counter() - started,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    write_json(args.output_dir / 'result.json', result)
    signal.alarm(0)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--objective', choices=['A', 'B'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args)
        print(json.dumps({'status': result['status'], 'objective': args.objective,
                          'wall_seconds': result['wall_seconds'], 'actual_calls': result['actual_calls']}))
    except Exception as error:
        # Preserve failures without printing transport/parser URL strings or changing prior evidence.
        if args.output_dir.exists():
            failure = {'status': 'STOP', 'failure_type': type(error).__name__}
            if type(error) is RuntimeError:
                failure['stop_reason'] = str(error)
            write_json(args.output_dir / 'failure.json', failure)
        print(json.dumps({'status': 'STOP', 'failure_type': type(error).__name__}))
        raise SystemExit(1)
