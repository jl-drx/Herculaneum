"""Independent phase19 toy review: no pretrained model, data or network IO."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-adapter-sha256', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    here = Path(__file__).resolve().parent
    source = here.parent / 'fusion-prototype/fusion_adapter.py'
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    assert source_sha == args.expected_adapter_sha256
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('reviewed_fusion_adapter', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []

    def check(name, condition, **details):
        assert condition, name
        checks.append(dict(name=name, passed=True, **details))

    def rejected(name, action):
        try:
            action()
        except ValueError:
            check(name, True)
        else:
            raise AssertionError(name)

    # NumPy loop is independent of the production slicing/catenation formula.
    rng = np.random.default_rng(190019)
    values = rng.normal(size=(2, 1, 17, 5, 7)).astype(np.float32)
    image = torch.from_numpy(values.copy())
    for arm in ('raw_control', 'explicit_depth'):
        reference = np.zeros((2, 3, 17, 5, 7), dtype=np.float32)
        for depth in range(17):
            left, right = max(0, depth-1), min(16, depth+1)
            if arm == 'raw_control':
                reference[:, 0, depth] = values[:, 0, left]
                reference[:, 1, depth] = values[:, 0, depth]
                reference[:, 2, depth] = values[:, 0, right]
            else:
                reference[:, 0, depth] = values[:, 0, depth]
                denominator = 1 if depth in (0, 16) else 2
                reference[:, 1, depth] = (values[:, 0, right] - values[:, 0, left]) / denominator
                reference[:, 2, depth] = (depth - 8) / 8
        actual = module.depth_features(image, arm).numpy()
        check(arm + '_independent_numpy_feature_bytes', np.array_equal(actual, reference))
        transformed = module.depth_features(image.flip(-1).transpose(-1, -2), arm)
        check(arm + '_spatial_flip_transpose_commute',
              torch.equal(transformed, module.depth_features(image, arm).flip(-1).transpose(-1, -2)))
    check('source_image_unchanged', np.array_equal(image.numpy(), values))
    rejected('image_grad_rejected', lambda: module.depth_features(image.clone().requires_grad_(), 'raw_control'))
    rejected('wrong_depth_rejected', lambda: module.depth_features(image[:, :, :16], 'raw_control'))
    rejected('wrong_dtype_rejected', lambda: module.depth_features(image.double(), 'raw_control'))
    rejected('empty_batch_rejected', lambda: module.depth_features(image[:0], 'raw_control'))
    bad = image.clone(); bad[0, 0, 0, 0, 0] = float('inf')
    rejected('infinite_input_rejected', lambda: module.depth_features(bad, 'raw_control'))
    overflow = image.clone()
    overflow[:, :, 0] = -torch.finfo(torch.float32).max
    overflow[:, :, 1] = torch.finfo(torch.float32).max
    rejected('derived_feature_overflow_rejected', lambda: module.depth_features(overflow, 'explicit_depth'))

    torch.manual_seed(112019)
    caller_rng = torch.get_rng_state().clone()
    adapters = [module.matched_adapter(20261008) for _ in range(2)]
    check('matched_initialization_preserves_caller_rng', torch.equal(caller_rng, torch.get_rng_state()))
    check('matched_initialization_all_parameters', all(torch.equal(a, b) for a, b in
        zip(adapters[0].parameters(), adapters[1].parameters())))
    check('independent_parameter_count', all(sum(p.numel() for p in a.parameters()) ==
        (3*8*27+8) + (16*32*9+32) == 5296 for a in adapters))

    class FeatureMap(nn.Module):
        def forward(self, x):
            return x.mean(2).expand(-1, 32, -1, -1).clone()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Conv2d(32, 3, 1), nn.BatchNorm2d(3),
                                        nn.Tanh(), nn.Conv2d(3, 1, 1))
        def forward(self, x):
            return {'ink': self.layers(x)}

    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_fusion = FeatureMap()
            self.network = Decoder()

    base = ToyBackbone().eval()
    before = {name: value.clone() for name, value in base.state_dict().items()}
    with torch.no_grad():
        fused = base.depth_fusion(image)
        teacher = base.network(fused)['ink']
    target = torch.linspace(-.7, .4, teacher.numel()).reshape_as(teacher)
    observations = {}
    for arm, adapter in zip(('raw_control', 'explicit_depth'), adapters):
        wrapper = module.FrozenFusionModel(base, adapter, arm).train()
        check(arm + '_train_preserves_eval_all_submodules', all(not item.training for item in base.modules()))
        check(arm + '_all_backbone_parameters_frozen', all(not p.requires_grad for p in base.parameters()))
        output, residual, adapted = wrapper.forward_from_fused(image, fused)
        check(arm + '_zero_residual_and_fused_identity', torch.count_nonzero(residual) == 0 and torch.equal(adapted, fused))
        check(arm + '_zero_projection_output_identity', torch.equal(output, teacher))
        check(arm + '_decoder_input_keeps_gradient_path', adapted.requires_grad and output.requires_grad)
        ((output - target).square().mean()).backward()
        first_stem = sum(float(p.grad.double().square().sum()) for p in adapter.stem.parameters())**.5
        first_projection = sum(float(p.grad.double().square().sum()) for p in adapter.projection.parameters())**.5
        check(arm + '_first_stem_gradient_exact_zero', first_stem == 0)
        check(arm + '_first_projection_gradient_finite_nonzero', first_projection > 0 and np.isfinite(first_projection))
        # Explicit update avoids importing an optimizer or reusing author fixture logic.
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.add_(parameter.grad, alpha=-.01)
                parameter.grad = None
        updated, _, _ = wrapper.forward_from_fused(image, fused)
        (updated - target).square().mean().backward()
        second_stem = sum(float(p.grad.double().square().sum()) for p in adapter.stem.parameters())**.5
        check(arm + '_second_stem_gradient_finite_nonzero', second_stem > 0 and np.isfinite(second_stem))
        check(arm + '_updated_projection_changes_output', not torch.equal(updated, teacher))
        check(arm + '_backbone_has_no_parameter_gradients', all(p.grad is None for p in base.parameters()))
        check(arm + '_every_parameter_and_buffer_unchanged', all(torch.equal(base.state_dict()[name], value) for name, value in before.items()))
        observations[arm] = dict(first_stem_norm=first_stem, first_projection_norm=first_projection,
                                 second_stem_norm=second_stem)
    with torch.inference_mode():
        inference_fused = fused.clone()
    rejected('inference_tensor_cache_rejected', lambda: wrapper.forward_from_fused(image, inference_fused))
    rejected('grad_requiring_cache_rejected', lambda: wrapper.forward_from_fused(image, fused.clone().requires_grad_()))
    base.network.layers[1].train()
    rejected('nested_decoder_training_mode_rejected', lambda: wrapper.forward_from_fused(image, fused))
    base.eval()
    base.depth_fusion.train()
    rejected('nested_fusion_training_mode_rejected_before_execution', lambda: wrapper.frozen_features(image))
    base.eval()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    check('review_resource_bounds', rss < 1024**3 and time.monotonic() - start < 120)
    check('audited_source_unchanged', hashlib.sha256(source.read_bytes()).hexdigest() == source_sha)
    result = dict(schema='phase19-independent-fusion-toy-v1', status='PASS',
        source=dict(path=str(source), sha256=source_sha, size_bytes=len(source_bytes)),
        check_count=len(checks), checks=checks, gradient_observations=observations,
        resources=dict(wall_seconds=time.monotonic()-start, peak_rss_bytes=rss,
                       threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads()),
        execution=dict(pretrained_checkpoint_loads=0, actual_seed42_constructions=0,
                       scientific_array_reads=0, network_calls=0, gpu_calls=0,
                       independent_toy_manual_parameter_updates=2, independent_toy_backwards=4),
        limitation='Independent feature arithmetic and toy autograd only; actual seed42 integration is not executed by this reviewer.')
    with (here / 'independent-toy-results-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'check_count', 'resources', 'execution')}))


if __name__ == '__main__':
    main()
