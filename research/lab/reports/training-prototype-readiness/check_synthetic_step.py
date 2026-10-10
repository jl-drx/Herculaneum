#!/usr/bin/env python3
"""One synthetic masked-BCE CPU step; never saves model outputs or weights."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(60)  # Default SIGALRM terminates the process even inside native code.
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2, 2 * 1024**2))
    def offline(*unused, **kwargs):
        raise ValueError('Synthetic mechanics check attempted network access')
    socket.socket.connect = offline; socket.socket.connect_ex = offline
    socket.socket.sendto = offline; socket.create_connection = offline; socket.getaddrinfo = offline
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
    output = args.output.resolve()
    require(not output.exists() and output.parent == Path(__file__).resolve().parent, 'Use a fresh local receipt path')
    require(sha(args.plan) == args.plan_sha256, 'Plan SHA differs')
    plan = json.loads(args.plan.read_text())
    require(plan['kind'] == 'synthetic-cpu-mechanics-v1' and plan['real_training_authorized'] is False, 'Wrong workload')
    for record in plan['verified_inputs']:
        require(Path(record['path']).stat().st_size == record['size_bytes'] and sha(record['path']) == record['sha256'], 'Input changed')
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(20261008)
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('verified_model_contract', plan['helper']['path'])
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    runtime = helper.load_runtime()
    model, metadata = helper.build_model(plan['model']['path'], plan['model'], runtime)
    model.train()
    image = torch.randn(1, 1, 17, 128, 128, dtype=torch.float32)
    y, x = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
    participation = ((y >= 16) & (y < 96) & (x >= 16) & (x < 96))[None, None]
    target = torch.full((1, 1, 128, 128), -1., dtype=torch.float32)  # -1 means unknown; never included in BCE.
    target[participation] = (y[participation[0, 0]] < 40).float()
    region = (x >= 48).long()[None, None]
    logits = model(image)['ink']
    if isinstance(logits, (list, tuple)): logits = logits[0]
    require(tuple(logits.shape) == (1, 1, 128, 128) and bool(torch.isfinite(logits).all()), 'Invalid logits')
    logits.retain_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[participation], target[participation])
    group_losses, groups = [], []
    for r in [0, 1]:
        for label in [0., 1.]:
            mask = participation & (region == r) & (target == label)
            require(bool(mask.any()), 'Each synthetic region needs positive and negative anchors')
            group_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], target[mask]))
            groups.append(dict(region=r, target=label, pixels=int(mask.sum())))
    anchor_loss = torch.stack(group_losses).mean()  # Equal class/region weights, mechanics only; not stepped.
    require(bool(torch.isfinite(loss)) and bool(torch.isfinite(anchor_loss)), 'Nonfinite illustrative losses')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # Exactly one plain masked BCE backward; no research-loss training.
    require(bool((logits.grad[~participation] == 0).all()), 'Unknown pixels affected logits gradient')
    require(bool((logits.grad[participation] != 0).any()), 'Known support gave no logits gradient')
    gradients = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    require(gradients and all(bool(torch.isfinite(p.grad).all()) for _, p in gradients), 'Invalid model gradients')
    changed_name, parameter = max(gradients, key=lambda item: float(item[1].grad.abs().max()))
    before = parameter.detach().clone()
    optimizer.step()
    changed = int(torch.count_nonzero(parameter.detach() != before))
    require(changed > 0 and all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'No finite parameter update')
    elapsed = time.perf_counter() - started
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    require(elapsed < 60 and rss <= 3 * 1024**2, 'Wall/RSS acceptance limit exceeded')
    result = dict(status='pass_synthetic_mechanics_only', seed=20261008, model_sha256=metadata['sha256'],
        strict_state_match=metadata['strict_state_match'], parameter_count=metadata['parameter_count'],
        input_shape=[1, 1, 17, 128, 128], positive_pixels=int((participation & (target == 1)).sum()),
        negative_pixels=int((participation & (target == 0)).sum()), unknown_pixels=int((~participation).sum()),
        masked_bce=float(loss.detach()), illustrative_balanced_anchor_bce=float(anchor_loss.detach()), anchor_groups=groups,
        unknown_logits_gradients_exactly_zero=True, finite_gradient_tensors=len(gradients),
        changed_parameter=changed_name, changed_elements=changed, optimizer='SGD lr0.001 momentum0, one step',
        wall_seconds=elapsed, peak_rss_kib=rss, threads=4, optimizer_steps=1,
        real_data_used=False, network_requests=0, weights_or_prediction_files_written=0,
        limitation='Synthetic forward/backward/step and masked-support mechanics only; no new method validation or production optimizer decision.')
    del optimizer, model, gradients, parameter, before, logits, loss, anchor_loss, group_losses
    gc.collect()  # Updated model exists only in this process; no checkpoint serialization.
    require(sha(args.plan) == args.plan_sha256 and sha(plan['model']['path']) == plan['model']['sha256'], 'Final identity changed')
    for record in plan['verified_inputs']: require(sha(record['path']) == record['sha256'], 'Final source changed')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    signal.alarm(0)
    print(json.dumps(dict(status=result['status'], receipt=str(output), wall_seconds=elapsed, peak_rss_kib=rss)))


if __name__ == '__main__': main()
