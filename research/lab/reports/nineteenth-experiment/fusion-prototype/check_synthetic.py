"""Synthetic fixtures only. No external model, checkpoint or scientific arrays."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import torch
from torch import nn
from fusion_adapter import (ARMS, PARAMETER_COUNT, FrozenFusionModel,
                            depth_features, matched_adapter)

HERE = Path(__file__).resolve().parent


def state_digest(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


class SyntheticFusion(nn.Module):
    def forward(self, image):
        mean = image.mean(dim=2)
        scales = torch.linspace(.5, 2., 32).view(1, 32, 1, 1)
        return mean * scales


class SyntheticBody(nn.Module):
    """Includes frozen affine normalization; this is not the released U-Net."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(32, 12, 3, padding=1),
            nn.InstanceNorm2d(12, affine=True), nn.LeakyReLU(.01),
            nn.Conv2d(12, 1, 1))
        self.register_buffer("unchanged_sentinel", torch.tensor([1., 2.]))

    def forward(self, fused):
        return {"ink": self.layers(fused)}


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_fusion = SyntheticFusion()
        self.network = SyntheticBody()

    def forward(self, image):
        return self.network(self.depth_fusion(image))


def main():
    start = time.monotonic()
    signal.alarm(300)
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(9182026)
    checks = []

    def check(name, passed, **details):
        if not passed:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True, **details})

    def rejects(name, action):
        try:
            action()
        except ValueError:
            check(name, True)
        else:
            raise AssertionError(name)

    # Analytical tests discriminate endpoint clamping from periodic/reflected
    # neighbors, full one-sided derivatives from half-slope endpoints, and
    # nominal index coordinates from acquired slice identifiers or distances.
    ramp = torch.arange(17, dtype=torch.float32).view(1, 1, 17, 1, 1).expand(2, 1, 17, 7, 9).clone()
    raw = depth_features(ramp, "raw_control")
    explicit = depth_features(ramp, "explicit_depth")
    check("linear_profile_derivative_including_endpoints", torch.equal(explicit[:, 1], torch.ones_like(explicit[:, 1])))
    check("coordinate_exact_endpoints_center_and_step", torch.equal(explicit[0, 2, :, 0, 0], (torch.arange(17) - 8) / 8))
    check("candidate_preserves_raw_center", torch.equal(explicit[:, :1], ramp))
    check("control_preserves_raw_center", torch.equal(raw[:, 1:2], ramp))
    check("previous_endpoint_clamped", torch.equal(raw[0, 0, :, 0, 0], torch.tensor([0., *range(16)])))
    check("next_endpoint_clamped", torch.equal(raw[0, 2, :, 0, 0], torch.tensor([*range(1, 17), 16.])))
    constant = torch.full_like(ramp, 3.)
    check("constant_profile_has_zero_derivative", not bool(depth_features(constant, "explicit_depth")[:, 1].any()))
    reverse = depth_features(ramp.flip(2), "explicit_depth")
    check("reversal_changes_derivative_not_nominal_coordinate", torch.equal(reverse[:, 1], -explicit[:, 1]) and torch.equal(reverse[:, 2], explicit[:, 2]))
    probe = torch.randn(2, 1, 17, 7, 9)
    for arm in ARMS:
        check(arm + "_xy_rotation_equivariance", torch.equal(depth_features(torch.rot90(probe, 1, (-2, -1)), arm), torch.rot90(depth_features(probe, arm), 1, (-2, -1))))
    rejects("wrong_depth_rejected", lambda: depth_features(probe[:, :, :16], "raw_control"))
    rejects("wrong_dtype_rejected", lambda: depth_features(probe.double(), "raw_control"))
    rejects("unknown_arm_rejected", lambda: depth_features(probe, "other"))
    bad = probe.clone(); bad[0, 0, 0, 0, 0] = float("nan")
    rejects("nonfinite_input_rejected", lambda: depth_features(bad, "raw_control"))

    rng_before = torch.get_rng_state().clone()
    adapters = {arm: matched_adapter() for arm in ARMS}
    check("initialization_preserves_caller_rng", torch.equal(rng_before, torch.get_rng_state()))
    check("arms_have_identical_initial_tensors", state_digest(adapters[ARMS[0]]) == state_digest(adapters[ARMS[1]]))
    check("matched_5296_parameters", all(sum(p.numel() for p in a.parameters()) == PARAMETER_COUNT for a in adapters.values()), parameters_per_arm=PARAMETER_COUNT)
    check("trainable_stem_nonzero_and_projection_zero", all(bool(a.stem.weight.any()) and not bool(a.projection.weight.any()) and not bool(a.projection.bias.any()) for a in adapters.values()))

    # One native-sized input verifies dimensional compatibility, independently
    # of the small fixture used to exercise reverse-mode gradients below.
    native = torch.randn(1, 1, 17, 128, 128)
    with torch.no_grad():
        for arm, adapter in adapters.items():
            result = adapter(depth_features(native, arm))
            check(arm + "_native_projection_shape_and_zero", result.shape == (1, 32, 128, 128) and not bool(result.any()))

    base = SyntheticBackbone().eval()
    base_before = state_digest(base)
    image = torch.randn(2, 1, 17, 24, 32)
    with torch.no_grad():
        baseline = base(image)["ink"]
        fused = base.depth_fusion(image)
    target = torch.linspace(-1, 1, baseline.numel()).reshape_as(baseline)
    gradient_observations = {}
    for arm, adapter in adapters.items():
        wrapper = FrozenFusionModel(base, adapter, arm).train()
        check(arm + "_train_keeps_all_backbone_modules_eval", all(not m.training for m in base.modules()))
        check(arm + "_backbone_parameters_frozen", all(not p.requires_grad for p in base.parameters()))
        optimizer = torch.optim.SGD(adapter.parameters(), lr=.01)
        check(arm + "_optimizer_contains_only_adapter", {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in adapter.parameters()})
        logits, residual, adapted = wrapper.forward_from_fused(image, fused)
        check(arm + "_zero_projection_exact_fused_identity", torch.equal(adapted, fused) and not bool(residual.any()))
        check(arm + "_zero_projection_exact_logit_identity", torch.equal(logits, baseline))
        loss = ((logits - target) ** 2).mean()
        loss.backward()
        first_projection = float(sum(p.grad.double().square().sum() for p in adapter.projection.parameters()).sqrt())
        first_stem = float(sum(p.grad.double().square().sum() for p in adapter.stem.parameters()).sqrt())
        check(arm + "_projection_gradient_crosses_frozen_decoder", first_projection > 0 and all(torch.isfinite(p.grad).all() for p in adapter.parameters()))
        check(arm + "_initial_hidden_gradient_is_exactly_zero", first_stem == 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updated_logits, _, _ = wrapper.forward_from_fused(image, fused)
        ((updated_logits - target) ** 2).mean().backward()
        second_stem = float(sum(p.grad.double().square().sum() for p in adapter.stem.parameters()).sqrt())
        check(arm + "_stem_gradient_after_projection_update", second_stem > 0 and torch.isfinite(torch.tensor(second_stem)))
        check(arm + "_backbone_never_receives_gradients", all(p.grad is None for p in base.parameters()))
        check(arm + "_backbone_state_and_buffers_unchanged", state_digest(base) == base_before)
        check(arm + "_updated_adapter_changes_logits", not torch.equal(updated_logits, baseline))
        gradient_observations[arm] = dict(first_projection_norm=first_projection,
            first_stem_norm=first_stem, second_stem_norm=second_stem)
    with torch.inference_mode():
        inference_fused = fused.clone()
    rejects("inference_mode_cache_rejected", lambda: wrapper.forward_from_fused(image, inference_fused))
    base.network.train()
    rejects("changed_backbone_mode_rejected", lambda: wrapper.forward_from_fused(image, fused))
    base.eval()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    check("resource_bound", time.monotonic() - start < 300 and max_rss < 4 * 1024**3)
    result = dict(kind="phase19-synthetic-fusion-prototype-checks-v1", status="PASS",
        check_count=len(checks), checks=checks, gradient_observations=gradient_observations,
        resources=dict(wall_seconds=time.monotonic()-start, peak_rss_bytes=max_rss,
                       threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads()),
        execution=dict(pretrained_checkpoint_loads=0, real_backbone_constructions=0,
            scientific_array_reads=0, network_calls=0, gpu_calls=0,
            toy_synthetic_optimizer_steps=2, toy_synthetic_backwards=4),
        limitation="Toy decoder only; this does not validate actual seed42 integration or scientific efficacy.")
    (HERE / "synthetic-results-v1.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "check_count", "resources", "execution")}))


if __name__ == "__main__":
    main()
