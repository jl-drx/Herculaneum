"""Independent small synthetic controls; no actual model or scientific arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE.parent / 'implementation'
sys.path.insert(0, str(IMPLEMENTATION))
from fusion_adapter import depth_features, matched_adapter, FrozenFusionModel
from depth_adapter import manual_terms


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(777)
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, callback):
        try:
            callback()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    image = torch.arange(17, dtype=torch.float32).square().reshape(1, 1, 17, 1, 1).expand(2, 1, 17, 5, 7).clone()
    raw = depth_features(image, 'raw_control')
    explicit = depth_features(image, 'explicit_depth')
    check('raw center exact', torch.equal(raw[:, 1:2], image))
    check('raw previous clamp', torch.equal(raw[:, 0:1, :1], image[:, :, :1]))
    check('raw following clamp', torch.equal(raw[:, 2:3, -1:], image[:, :, -1:]))
    check('raw interior shift', torch.equal(raw[:, 0:1, 1:], image[:, :, :-1]) and torch.equal(raw[:, 2:3, :-1], image[:, :, 1:]))
    check('explicit center exact', torch.equal(explicit[:, :1], image))
    check('quadratic centered interior differences', torch.equal(explicit[:, 1:2, 1:-1], (2 * torch.arange(1, 16, dtype=torch.float32)).reshape(1, 1, 15, 1, 1).expand(2, 1, 15, 5, 7)))
    check('full one-sided endpoints', bool((explicit[:, 1, 0] == 1).all()) and bool((explicit[:, 1, -1] == 31).all()))
    check('coordinate endpoint and origin', bool((explicit[:, 2, 0] == -1).all()) and bool((explicit[:, 2, 8] == 0).all()) and bool((explicit[:, 2, -1] == 1).all()))
    rejects('unknown feature arm rejected', lambda: depth_features(image, 'unexpected'))
    rejects('non-FP32 input rejected', lambda: depth_features(image.double(), 'raw_control'))
    rejects('trainable input rejected', lambda: depth_features(image.clone().requires_grad_(), 'raw_control'))
    rejects('wrong depth rejected', lambda: depth_features(image[:, :, :16], 'raw_control'))
    rejects('empty spatial input rejected', lambda: depth_features(image[:, :, :, :0], 'raw_control'))
    bad = image.clone(); bad.flatten()[0] = float('nan')
    rejects('nonfinite input rejected', lambda: depth_features(bad, 'raw_control'))
    rng = torch.random.get_rng_state().clone()
    adapters = [matched_adapter(), matched_adapter()]
    check('dedicated initializer preserves caller RNG', torch.equal(rng, torch.random.get_rng_state()))
    check('matched exact initial tensors', all(torch.equal(a, b) for a, b in zip(adapters[0].parameters(), adapters[1].parameters())))
    check('each adapter has 5296 parameters', all(sum(p.numel() for p in a.parameters()) == 5296 for a in adapters))
    check('projection begins exactly zero', all(not bool(p.any()) for a in adapters for p in a.projection.parameters()))

    class SyntheticDepth(nn.Module):
        def forward(self, value):
            return value.mean(2).expand(-1, 32, -1, -1).clone()

    class SyntheticBody(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 1, 1)
            self.norm = nn.BatchNorm2d(1)

        def forward(self, value):
            return {'ink': self.norm(self.conv(value)).tanh()}

    class SyntheticBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_fusion = SyntheticDepth()
            self.network = SyntheticBody()

        def forward(self, value):
            return self.network(self.depth_fusion(value))

    image = image / 256
    for arm, adapter in zip(('raw_control', 'explicit_depth'), adapters):
        model = SyntheticBackbone()
        wrapper = FrozenFusionModel(model, adapter, arm).train()
        check(arm + ': train() preserves all backbone eval modes', all(not m.training for m in model.modules()))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        fused = wrapper.frozen_features(image)
        check(arm + ': ordinary detached fusion cache', not fused.requires_grad and not torch.is_inference(fused))
        with torch.no_grad(): original = model(image)['ink']
        logits, residual, adapted = wrapper.forward_from_fused(image, fused)
        check(arm + ': exact initial feature and logit identity', torch.equal(logits, original) and torch.equal(adapted, fused) and not bool(residual.any()))
        target = torch.linspace(-.8, .8, logits.numel()).reshape_as(logits)
        loss = (logits - target).square().mean()
        loss.backward()
        check(arm + ': graph traverses frozen decoder', all(p.grad is not None for p in adapter.parameters()) and all(p.grad is None for p in model.parameters()))
        check(arm + ': zero initial stem gradient, nonzero projection', all(not bool(p.grad.any()) for p in adapter.stem.parameters()) and any(bool(p.grad.any()) for p in adapter.projection.parameters()))
        optimizer = torch.optim.SGD(adapter.parameters(), lr=.001)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        second, _, _ = wrapper.forward_from_fused(image, fused)
        (second - target).square().mean().backward()
        check(arm + ': second gradient reaches stem', any(bool(p.grad.any()) for p in adapter.stem.parameters()))
        check(arm + ': backbone parameters and buffers immutable', all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        rejects(arm + ': trainable cache rejected', lambda: wrapper.forward_from_fused(image, fused.clone().requires_grad_()))
        rejects(arm + ': wrong cache channel count rejected', lambda: wrapper.forward_from_fused(image, fused[:, :31]))
        with torch.inference_mode(): inference_cache = fused.clone()
        rejects(arm + ': inference-mode cache rejected', lambda: wrapper.forward_from_fused(image, inference_cache))
        model.network.train()
        rejects(arm + ': changed decoder mode rejected', lambda: wrapper.forward_from_fused(image, fused))

    logits = torch.tensor([-.9, .4, 1.7, -.2, 2.4, .8], dtype=torch.float32).reshape(1, 1, 2, 3).requires_grad_()
    teacher = torch.tensor([.2, -.8, 1.5, .1, 2.1, -.7], dtype=torch.float32).reshape_as(logits).requires_grad_()
    mask = torch.tensor([True, False, True, True, False, True]).reshape_as(logits)
    target = torch.tensor([0, 999, 1, 0, -999, 1], dtype=torch.float32).reshape_as(logits)
    loss, terms = manual_terms(logits, teacher, mask, target)
    loss.backward()
    expected = torch.zeros_like(logits)
    expected[mask] = (2 * logits.detach()[mask].double().sigmoid() - teacher.detach()[mask].double().sigmoid() - (.25 + .5 * target[mask].double())) / int(mask.sum())
    check('manual logit derivative', torch.allclose(logits.grad, expected, rtol=0, atol=2e-8))
    check('unknown target values excluded from objective', bool((logits.grad[~mask] == 0).all()))
    check('teacher has no gradient', teacher.grad is None)
    check('reported total equals BCE plus KL', abs(terms['full'] - terms['base'] - terms['kl']) < 1e-12)
    output = dict(status='PASS', kind='phase20-independent-small-synthetic-controls',
                  checks=checks, check_count=len(checks),
                  source_pins={name: pin(IMPLEMENTATION / name) for name in ['fusion_adapter.py', 'depth_adapter.py', 'depth_common.py']},
                  fixture_source=pin(Path(__file__).resolve()), actual_checkpoint_reads=0,
                  scientific_array_reads=0, actual_model_executions=0, network_calls=0,
                  synthetic_optimizer_steps=2,
                  limitations=['Finite synthetic controls and source review, not independent convolution-autograd proof or actual-data efficacy.'])
    with (HERE / 'synthetic-controls-v1.json').open('x') as stream:
        json.dump(output, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps(dict(status='PASS', check_count=len(checks))))


if __name__ == '__main__':
    main()
