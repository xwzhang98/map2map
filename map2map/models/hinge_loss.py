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
        # d_loss_fake = F.relu(1.0 + D_fake).mean(dim=[0, 1])
        # d_loss_real = F.relu(1.0 - D_real).mean(dim=[0, 1])
        d_loss_fake = F.relu(1.0 + D_fake).mean()
        d_loss_real = F.relu(1.0 - D_real).mean()
        return d_loss_real, d_loss_fake


# def hinge_grad_penalty(critic, x, y, lam=10, *args, **kwargs):
#     """Calculate the gradient penalty for WGAN"""
#     device = x.device
#     batch_size = x.shape[0]
#     alpha = torch.rand(batch_size, device=device)
#     alpha = alpha.reshape(batch_size, *(1,) * (x.dim() - 1))

#     xy = alpha * x.detach() + (1 - alpha) * y.detach()

#     score = critic(xy.requires_grad_(True), *args, **kwargs)
#     # average over spatial dimensions if present
#     score = score.flatten(start_dim=1).mean(dim=1)
#     # sum over batches because graphs are mostly independent (w/o batchnorm)
#     score = score.sum()

#     (grad,) = torch.autograd.grad(
#         score,
#         xy,
#         retain_graph=True,
#         create_graph=True,
#         only_inputs=True,
#     )

#     grad = grad.flatten(start_dim=1)
#     penalty = (
#         lam * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
#         + 0 * score  # hack to trigger DDP allreduce hooks
#     )

#     return penalty

def hinge_grad_penalty(critic, x, y, lam=10, *args, **kwargs):
    """Calculate the L-infinity gradient penalty with Hinge function
    
    Penalizes the model only when ||∇f(x)||∞ > 1 using max(0, ||∇f(x)||∞ - 1)
    This corresponds to maximizing the L1-norm margin as described in the paper.
    """
    device = x.device
    batch_size = x.shape[0]
    alpha = torch.rand(batch_size, device=device)
    alpha = alpha.reshape(batch_size, *(1,) * (x.dim() - 1))

    # Interpolation between real and fake samples
    xy = alpha * x.detach() + (1 - alpha) * y.detach()
    xy.requires_grad_(True)

    score = critic(xy, *args, **kwargs)
    # Average over spatial dimensions if present
    score = score.flatten(start_dim=1).mean(dim=1)
    # Sum over batches
    score = score.sum()

    (grad,) = torch.autograd.grad(
        score,
        xy,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )

    # Compute L-infinity norm (maximum absolute value of any component)
    grad = grad.flatten(start_dim=1)
    
    # L-infinity norm is the maximum absolute value in each sample's gradient
    grad_norm_inf = grad.abs().max(dim=1)[0]
    
    # Hinge penalty: only penalize when norm > 1
    # Using max(0, ||∇f(x)||∞ - 1) which is the recommended function from the paper
    penalty = lam * F.relu(grad_norm_inf - 1).mean()
    
    # Add DDP hack to trigger allreduce hooks
    penalty = penalty + 0 * score
    
    return penalty, grad_norm_inf


def r1_regularization(critic, real_data, style=None, lam=10.0):
    """Memory-optimized R1 regularization for StyleGAN2.
    
    This penalizes the gradient norm at real data points only.
    """
    real_data = real_data.detach().requires_grad_(True)
    
    if style is not None:
        real_pred = critic(real_data, style)
    else:
        real_pred = critic(real_data)
    
    # Average over spatial dimensions if present and sum for scalar output
    real_pred = real_pred.flatten(start_dim=1).mean(dim=1).sum()
    
    grad_real = torch.autograd.grad(
        outputs=real_pred,
        inputs=real_data,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute R1 gradient penalty with memory optimization
    # Use in-place operations where possible
    grad_flat = grad_real.flatten(start_dim=1)
    r1_penalty = lam * 0.5 * grad_flat.square().sum(1).mean()
    
    # Clear intermediate tensors immediately
    del grad_real, grad_flat
    
    # DDP hack
    r1_penalty = r1_penalty + 0 * real_pred
    
    # Force memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return r1_penalty