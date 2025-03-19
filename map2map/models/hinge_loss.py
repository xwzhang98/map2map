import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    """Wasserstein distance

    target should have values of 0 (False) or 1 (True)
    """

    def __init__(self):
        super().__init__()

    def forward(self, D_fake, D_real=None):
        if D_real is None:
            return self._forward_G(D_fake)
        else:
            return self._forward_D(D_fake, D_real)

    def _forward_G(self, D_fake):
        return -D_fake.mean()

    def _forward_D(self, D_fake, D_real):
        # hinge loss
        d_loss_fake = F.relu(1.0 + D_fake).mean()
        d_loss_real = F.relu(1.0 - D_real).mean()
        return d_loss_real, d_loss_fake


def hinge_grad_penalty(critic, x, y, lam=10, *args, **kwargs):
    """Calculate the gradient penalty for WGAN"""
    device = x.device
    batch_size = x.shape[0]
    alpha = torch.rand(batch_size, device=device)
    alpha = alpha.reshape(batch_size, *(1,) * (x.dim() - 1))

    xy = alpha * x.detach() + (1 - alpha) * y.detach()

    score = critic(xy.requires_grad_(True), *args, **kwargs)
    # average over spatial dimensions if present
    score = score.flatten(start_dim=1).mean(dim=1)
    # sum over batches because graphs are mostly independent (w/o batchnorm)
    score = score.sum()

    (grad,) = torch.autograd.grad(
        score,
        xy,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )

    grad = grad.flatten(start_dim=1)
    penalty = (
        lam * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
        + 0 * score  # hack to trigger DDP allreduce hooks
    )

    return penalty
