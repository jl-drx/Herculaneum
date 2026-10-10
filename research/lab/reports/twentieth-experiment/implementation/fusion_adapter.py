"""Matched synthetic-validated fusion adapters; no artifact or network IO."""
from __future__ import annotations

import torch
from torch import nn

ARMS = ("raw_control", "explicit_depth")
PARAMETER_COUNT = 5296


def require(ok, message):
    if not ok:
        raise ValueError(message)


def image_contract(image):
    require(image.ndim == 5 and image.shape[1:3] == (1, 17)
            and image.shape[0] > 0 and min(image.shape[3:]) > 0
            and image.dtype == torch.float32 and image.device.type == "cpu"
            and not image.requires_grad and bool(torch.isfinite(image).all()),
            "Expected finite detached CPU FP32 [B,1,17,H,W]")


def depth_features(image, arm):
    """Preserve published depth order; r is nominal index, not physical distance."""
    image_contract(image)
    require(arm in ARMS, "Unknown representation arm")
    if arm == "raw_control":
        previous = torch.cat((image[:, :, :1], image[:, :, :-1]), dim=2)
        following = torch.cat((image[:, :, 1:], image[:, :, -1:]), dim=2)
        return torch.cat((previous, image, following), dim=1)
    derivative = torch.empty_like(image)
    derivative[:, :, 1:-1] = (image[:, :, 2:] - image[:, :, :-2]) * .5
    derivative[:, :, :1] = image[:, :, 1:2] - image[:, :, :1]
    derivative[:, :, -1:] = image[:, :, -1:] - image[:, :, -2:-1]
    coordinate = ((torch.arange(17, dtype=torch.float32) - 8) / 8).view(1, 1, 17, 1, 1)
    require(bool(torch.isfinite(derivative).all()), "Nonfinite derived depth feature")
    return torch.cat((image, derivative, coordinate.expand_as(image)), dim=1)


class FusionAdapter(nn.Module):
    """3→8 local features, mean/max depth pooling, zero 16→32 projection."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(3, 8, kernel_size=3, padding=1, bias=True)
        self.activation = nn.LeakyReLU(.01, inplace=False)
        self.projection = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        require(sum(p.numel() for p in self.parameters()) == PARAMETER_COUNT,
                "Adapter capacity changed")

    def forward(self, representation):
        hidden = self.activation(self.stem(representation))
        pooled = torch.cat((hidden.mean(dim=2), hidden.amax(dim=2)), dim=1)
        return self.projection(pooled)


def matched_adapter(seed=20261008):
    """Dedicated repeatable initialization without consuming caller RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FusionAdapter()


class FrozenFusionModel(nn.Module):
    """Keep original model immutable; differentiate its 2D body with respect to input.

    `model` is the already verified Local3DStem2DUNet, or a synthetic interface
    fixture. This class does not load, validate or substitute a checkpoint.
    """
    def __init__(self, model, adapter, arm):
        super().__init__()
        require(arm in ARMS, "Unknown representation arm")
        require(hasattr(model, "depth_fusion") and hasattr(model, "network"),
                "Missing original fusion/body interface")
        self.backbone = model
        self.adapter = adapter
        self.arm = arm
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def frozen_features(self, image):
        image_contract(image)
        require(not self.backbone.training
                and not any(m.training for m in self.backbone.modules())
                and not any(p.requires_grad for p in self.backbone.parameters()),
                "Backbone must remain frozen and in eval mode")
        with torch.no_grad():
            fused = self.backbone.depth_fusion(image)
        # no_grad yields ordinary tensors, unlike inference_mode caches.
        return fused.detach()

    def forward_from_fused(self, image, fused):
        image_contract(image)
        require(not self.backbone.training
                and not any(m.training for m in self.backbone.modules())
                and not any(p.requires_grad for p in self.backbone.parameters()),
                "Backbone must remain frozen and every module in eval mode")
        require(fused.shape == (image.shape[0], 32, *image.shape[3:])
                and fused.dtype == torch.float32 and fused.device.type == "cpu"
                and not fused.requires_grad and not torch.is_inference(fused)
                and bool(torch.isfinite(fused).all()),
                "Expected detached ordinary finite FP32 32-channel fused features")
        residual = self.adapter(depth_features(image, self.arm))
        adapted_features = fused + residual
        # DO NOT wrap this call in no_grad: frozen parameters still permit
        # an input gradient, which is the adapter's complete learning path.
        output = self.backbone.network(adapted_features)
        require(isinstance(output, dict) and "ink" in output
                and output["ink"].shape == (image.shape[0], 1, *image.shape[3:])
                and bool(torch.isfinite(output["ink"]).all()), "Invalid decoder output")
        return output["ink"], residual, adapted_features

    def forward(self, image):
        return self.forward_from_fused(image, self.frozen_features(image))
