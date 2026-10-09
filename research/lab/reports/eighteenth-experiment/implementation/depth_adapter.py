"""Matched 801-parameter residual adapters; no checkpoint or scientific input IO."""
import torch
from torch import nn
from depth_common import require

ARMS = ['raw_control', 'explicit_depth']


def features(image, arm):
    require(arm in ARMS, 'Unknown representation arm')
    require(image.ndim == 5 and image.shape[1:3] == (1, 17)
            and image.dtype == torch.float32 and image.device.type == 'cpu'
            and bool(torch.isfinite(image).all()), 'Expected finite CPU FP32 [B,1,17,H,W]')
    if arm == 'raw_control':
        previous = torch.cat((image[:, :, :1], image[:, :, :-1]), dim=2)
        following = torch.cat((image[:, :, 1:], image[:, :, -1:]), dim=2)
        result = torch.cat((previous, image, following), dim=1)
    else:
        derivative = torch.empty_like(image)
        derivative[:, :, 1:-1] = (image[:, :, 2:] - image[:, :, :-2]) * .5
        derivative[:, :, :1] = image[:, :, 1:2] - image[:, :, :1]
        derivative[:, :, -1:] = image[:, :, -1:] - image[:, :, -2:-1]
        coordinate = ((torch.arange(17, dtype=torch.float32) - 8) / 8).view(1, 1, 17, 1, 1)
        result = torch.cat((image, derivative, coordinate.expand_as(image)), dim=1)
    require(result.dtype == torch.float32 and bool(torch.isfinite(result).all()), 'Nonfinite features')
    return result


class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(3, 8, kernel_size=3, padding=1, bias=True)
        self.activation = nn.LeakyReLU(negative_slope=.01, inplace=False)
        self.head = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        require(sum(p.numel() for p in self.parameters()) == 801, 'Adapter capacity changed')

    def forward(self, representation):
        values = self.activation(self.stem(representation))
        pooled = torch.cat((values.mean(dim=2), values.amax(dim=2)), dim=1)
        residual = self.head(pooled)
        require(residual.dtype == torch.float32 and bool(torch.isfinite(residual).all()), 'Invalid residual')
        return residual


def compose(adapter, representation, frozen_logits):
    require(not frozen_logits.requires_grad and frozen_logits.grad is None,
            'Cached backbone logits must have no gradient')
    residual = adapter(representation)
    require(residual.shape == frozen_logits.shape and frozen_logits.dtype == torch.float32
            and bool(torch.isfinite(frozen_logits).all()), 'Frozen logit contract differs')
    logits = frozen_logits + residual
    require(bool(torch.isfinite(logits).all()), 'Nonfinite composed logits')
    return logits, residual


def manual_terms(logits, teacher, mask, hard_target):
    """FP32 epsilon=.5 manual BCE plus FP64 KL(teacher || student), unit weight."""
    require(mask.dtype == torch.bool and mask.shape == logits.shape == teacher.shape == hard_target.shape
            and bool(mask.any()) and bool(((hard_target[mask] == 0) | (hard_target[mask] == 1)).all()),
            'Manual participation/target contract differs')
    soft = .25 + .5 * hard_target[mask]
    base = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], soft)
    z = logits[mask].double()
    t = teacher.detach()[mask].double()
    divergence = torch.logaddexp(z, torch.zeros_like(z)) - torch.logaddexp(t, torch.zeros_like(t)) - torch.sigmoid(t) * (z - t)
    kl = divergence.mean()
    total = base + kl
    require(bool(torch.isfinite(total)) and float(kl.detach()) >= -1e-6, 'Nonfinite/negative KL loss')
    with torch.no_grad():
        hard = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], hard_target[mask])
        drift = (torch.sigmoid(z) - torch.sigmoid(t)).abs().mean()
    return total, dict(base=float(base.detach()), full=float(total.detach()), kl=float(kl.detach()),
                      hard_bce=float(hard), probability_drift_manual=float(drift))
