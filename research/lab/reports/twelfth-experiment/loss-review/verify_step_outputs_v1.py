"""NumPy-only verification of saved first-step logits and update summaries."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent/'step-probe'


def record(path):
    path = Path(path).resolve(strict=True)
    assert path.suffix not in ['.pth', '.pt'], 'Checkpoint reads forbidden'
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def verify(pin):
    actual = record(pin['path'])
    assert all(actual[k] == pin[k] for k in actual)
    return Path(pin['path'])


def close(a, b, atol=2e-12):
    assert abs(a-b) <= atol, (a, b, atol)


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def support(z, initial, target):
    z, initial, target = [np.asarray(v, dtype=np.float64) for v in [z, initial, target]]
    smooth = .25+.5*target
    a, a0 = np.logaddexp(0., z), np.logaddexp(0., initial)
    q = sigmoid(initial)
    k = np.mean(a-a0-q*(z-initial))
    linear = np.mean((q-smooth)*(z-initial))
    delta = np.mean((a-smooth*z)-(a0-smooth*initial))
    hard_delta = np.mean((a-target*z)-(a0-target*initial))
    return dict(smooth=float(np.mean(a-smooth*z)), hard=float(np.mean(a-target*z)),
                kl=float(k), linear=float(linear), base_delta=float(delta), hard_delta=float(hard_delta))


def main():
    index_path = PROBE/'run/run-results.json'
    assert record(index_path)['sha256'] == 'afae91cf22d96ab305e6f74505a9fc0848fd33bfc482c6e0a6494bacc5f2ee4c'
    index = json.loads(index_path.read_text())
    plan = json.loads(verify(index['plan']).read_text())
    assert index['plan']['sha256'] == 'd57d1a9d17bc11d8dd6630de58deb73338674e91530dc522ff28184a7afc599d'
    assert index['status'] == 'PASS' and index['retries'] == 0 and index['wall_seconds'] <= 300
    verify(index['controller'])
    for pin in plan['code'].values():
        verify(pin)
    assert [w['objective'] for w in index['workers']] == ['A', 'B']
    assert sum(p.stat().st_size for p in (PROBE/'run').rglob('*') if p.is_file()) < 64*1024**2
    checks, losses = [], []
    parameter_rows = 0
    maximum_loss_error = maximum_stable_error = maximum_probe_ratio = 0.
    prior_inputs = None
    for worker in index['workers']:
        assert worker['returncode'] == 0 and not worker['timed_out'] and worker['wall_seconds'] <= 125
        result = json.loads(verify(worker['result']).read_text())
        arm = result['objective']
        coefficient = 0. if arm == 'A' else .25
        assert result['status'] == 'PASS' and result['plan'] == index['plan']
        assert result['initial_state_sha256'] == plan['initial_state_sha256']
        assert result['actual_calls'] == dict(model_forwards=4, backwards=1, independent_fresh_optimizer_steps=3)
        assert result['wall_seconds'] <= 120 and result['peak_rss_bytes'] < 3*1024**3
        assert result['torch_version'] == plan['runtime']['torch']
        for pin in result['outputs']:
            verify(pin)
        directory = Path(worker['result']['path']).parent
        arrays = {name: np.load(directory/(file+'.npy'), allow_pickle=False) for name, file in {
            'input': 'input', 'manual_mask': 'manual-mask', 'manual_hard_target': 'manual-hard-target',
            'pseudo_mask': 'pseudo-mask', 'pseudo_hard_target': 'pseudo-hard-target'}.items()}
        digest = hashlib.sha256()
        for key in sorted(arrays):
            array = arrays[key]
            dtype = 'torch.bool' if array.dtype == np.bool_ else 'torch.float32'
            assert array.dtype in [np.dtype('float32'), np.dtype('bool')]
            digest.update(json.dumps([key, list(array.shape), dtype], separators=(',', ':')).encode()+b'\0')
            digest.update(np.ascontiguousarray(array).tobytes())
        assert digest.hexdigest() == result['input_target_sha256'] == '571d10c58008cb6f9e581a717bbb067846dae6a2ee2d29bdf9eb141ca9a02ec3'
        if prior_inputs is not None:
            for key in arrays:
                assert np.array_equal(arrays[key], prior_inputs[key])
        prior_inputs = arrays
        m, p = arrays['manual_mask'], arrays['pseudo_mask']
        y, h = arrays['manual_hard_target'], arrays['pseudo_hard_target']
        assert m.dtype == p.dtype == np.bool_ and not (m & p).any()
        assert m.shape == p.shape == y.shape == h.shape == (2, 1, 128, 128)
        assert np.all((y[m] == 0) | (y[m] == 1)) and np.all((h[p] == 0) | (h[p] == 1))
        initial = np.load(verify(result['initial_logits']), allow_pickle=False)
        assert initial.shape == m.shape and initial.dtype == np.float32
        for pin in plan['historical'][arm]['initial_logits'].values():
            archived = np.load(verify(pin), mmap_mode='r', allow_pickle=False)
            assert np.array_equal(np.ascontiguousarray(archived[0]).view(np.uint32), initial.view(np.uint32))
        expected_clip = float(np.minimum(np.float32(1), np.float32(1)/(np.float32(result['preclip_global_gradient_norm'])+np.float32(1e-6))))
        assert expected_clip == result['actual_clip_coefficient']
        assert [t['learning_rate'] for t in result['trials']] == [1e-6, 1e-5, 1e-4]
        for trial in result['trials']:
            eta = trial['learning_rate']
            assert trial['initial_state_sha256'] == plan['initial_state_sha256']
            assert trial['clipped_gradient_sha256'] == result['clipped_gradient_sha256']
            assert trial['optimizer_initial_state_empty'] and trial['optimizer_step'] == 1
            post = np.load(verify(trial['post_logits']), allow_pickle=False)
            assert post.shape == initial.shape and post.dtype == np.float32 and np.isfinite(post).all()
            terms = {}
            for label, mask, target in [('manual', m, y), ('pseudo', p, h)]:
                terms[label] = support(post[mask], initial[mask], target[mask])
                for key in ['kl', 'linear', 'base_delta', 'hard_delta']:
                    error = abs(terms[label][key]-trial['stable_change'][label][key])
                    maximum_stable_error = max(maximum_stable_error, error)
                    assert error <= 1e-12
            weighted = {key: terms['manual'][key]+coefficient*terms['pseudo'][key]
                        for key in ['kl', 'linear', 'base_delta', 'hard_delta']}
            weighted['full_delta'] = weighted['base_delta']+weighted['kl']
            for key, value in weighted.items():
                close(value, trial['stable_change']['weighted'][key])
            close(weighted['base_delta'], weighted['kl']+weighted['linear'])
            for name, z in [('pre', initial), ('post', post)]:
                tm, tp = support(z[m], initial[m], y[m]), support(z[p], initial[p], h[p])
                balanced = np.mean([support(z[r][m[r] & (y[r] == hard)], initial[r][m[r] & (y[r] == hard)],
                    y[r][m[r] & (y[r] == hard)])['smooth'] for r in range(2) for hard in [0., 1.]])
                expected = dict(smooth_U_fp32=tm['smooth'], smooth_P_fp32=tp['smooth'], smooth_R_fp32=float(balanced),
                    base_fp32=tm['smooth']+coefficient*tp['smooth'], kl_M_fp64=tm['kl'], kl_P_fp64=tp['kl'],
                    weighted_kl_fp64=tm['kl']+coefficient*tp['kl'],
                    full_mixed_dtype=tm['smooth']+coefficient*tp['smooth']+tm['kl']+coefficient*tp['kl'],
                    hard_U_fp32=tm['hard'], hard_P_fp32=tp['hard'], hard_base_fp32=tm['hard']+coefficient*tp['hard'])
                for key, value in expected.items():
                    error = abs(value-trial[name][key])
                    maximum_loss_error = max(maximum_loss_error, error)
                    assert error <= (1e-12 if key in ['kl_M_fp64', 'kl_P_fp64', 'weighted_kl_fp64'] else 2e-6)
            update = json.loads(verify(trial['parameter_update_summary']).read_text())
            rows = [r for r in update['per_parameter'] if r.get('gradient', 'present') is not None]
            parameter_rows += len(rows)
            assert len({r['name'] for r in rows}) == len(rows)
            assert update['wrong_sign_coordinates'] == 0
            assert all(r['optimizer_step'] == 1 and r['gradient_dot_actual_update'] <= 0
                       and r['max_error_allowance_ratio'] <= 1 for r in rows)
            for name in ['actual_update_l2', 'ideal_update_l2']:
                close(math.sqrt(sum(r[name]**2 for r in rows)), update[name], 1e-10)
            close(sum(r['gradient_dot_actual_update'] for r in rows), update['gradient_dot_actual_update'])
            close(math.sqrt(sum(r['clipped_gradient_l2']**2 for r in rows)), result['clipped_global_gradient_norm_fp64'])
            assert sum(r['changed_coordinates'] for r in rows) == update['changed_coordinates']
            assert sum(r['numel']-r['changed_coordinates'] for r in rows) == update['unchanged_coordinates']
            assert max(r['max_absolute_identity_error'] for r in rows) == update['max_absolute_identity_error']
            assert max(r['max_error_allowance_ratio'] for r in rows) == update['max_error_allowance_ratio']
            for row in rows:
                probe = row['probe']
                g = probe['clipped_gradient']
                ideal = -eta*g/(abs(g)+1e-8)
                close(ideal, probe['ideal_delta'], 1e-16)
                actual = probe['after']-probe['before']
                before = np.float32(probe['before'])
                ulp = abs(float(np.nextafter(before, np.float32(np.inf), dtype=np.float32)-before))
                allowance = 4*ulp+8*np.finfo(np.float32).eps*abs(ideal)+1e-12
                ratio = abs(actual-ideal)/allowance
                assert ratio <= 1 and g*actual <= 0
                maximum_probe_ratio = max(maximum_probe_ratio, ratio)
            close(trial['gradient_dot_actual_update'], update['gradient_dot_actual_update'])
            assert trial['gradient_dot_actual_update'] < 0
            assert trial['max_first_adam_error_allowance_ratio'] == update['max_error_allowance_ratio']
            losses.append(dict(objective=arm, learning_rate=eta, **weighted))
        checks.append(dict(objective=arm, input_hash=result['input_target_sha256'],
                           clipped_gradient_sha256=result['clipped_gradient_sha256']))
    target = HERE/'output-verification'
    target.mkdir(exist_ok=False)
    result = dict(status='PASS', plan=index['plan'], run_index=record(index_path), checker=record(Path(__file__)),
        branches_verified=6, parameter_summary_rows_verified=parameter_rows, input_checks=checks,
        reconstructed_changes=losses, maximum_FP32_loss_reconstruction_error=maximum_loss_error,
        maximum_FP64_stable_term_error=maximum_stable_error, maximum_selected_parameter_probe_allowance_ratio=maximum_probe_ratio,
        parameter_verification_scope='Aggregated reported parameter summaries and one retained coordinate per gradient-bearing tensor; not independent replay of every parameter coordinate, because raw gradients/full updated states were not retained.',
        model_calls=0, checkpoint_reads=0, raw_gradient_reads=0, network_calls=0)
    path = target/'verification.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', receipt=record(path))))


if __name__ == '__main__':
    main()
