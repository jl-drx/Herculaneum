"""Exact architecture/loss extracted from the frozen CPU study; no training state."""
import torch
from torch import nn
import torch.nn.functional as F

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
