"""
Muon Optimizer — for 2D weight matrices in neural networks.

Based on: "Muon: An optimizer for hidden layers" (Bernstein et al.)
Reference: https://github.com/KellerJordan/Muon

Key idea: instead of standard SGD/Adam updates, Muon applies a
"spectrally normalized" update using Newton-Schulz iterations to
approximate the polar decomposition of the gradient. This makes
updates direction-only (orthogonal), which is better for training
weight matrices in transformers.

Usage:
    - Apply Muon to 2D weight matrices (attention Q/K/V/O, MLP up/down)
    - Use AdamW for 1D parameters (embeddings, layernorm, lm_head)
    - This split is standard practice with Muon
"""

import torch
from torch.optim import Optimizer


def _newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Approximate the polar decomposition G = U @ S @ V^T -> U @ V^T
    using Newton-Schulz iterations.

    This gives us the "direction" of the gradient in matrix space,
    stripping out the magnitude (singular values).

    The iteration: X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k^T @ X_k
    converges to the orthogonal polar factor.
    """
    assert G.ndim == 2, "Newton-Schulz only works on 2D matrices"

    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimized coefficients

    # Transpose if tall matrix (more rows than cols)
    transposed = False
    if G.shape[0] > G.shape[1]:
        G = G.T
        transposed = True

    # Normalize so spectral norm ≈ 1
    spectral_norm = G.norm()
    if spectral_norm < eps:
        return torch.zeros_like(G.T if transposed else G)
    X = G / spectral_norm

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X


class Muon(Optimizer):
    """
    Muon optimizer for 2D weight matrices.

    Args:
        params: Parameters to optimize (should be 2D weight matrices only)
        lr: Learning rate (default: 0.02)
        momentum: Nesterov momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        weight_decay: L2 weight decay (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # Weight decay (decoupled, like AdamW)
                if wd > 0:
                    p.mul_(1 - lr * wd)

                state = self.state[p]

                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]

                # Momentum update
                buf.mul_(momentum).add_(g)

                if nesterov:
                    # Nesterov: use g + momentum * buf as the effective gradient
                    g_eff = g + momentum * buf
                else:
                    g_eff = buf

                # Apply Newton-Schulz orthogonalization to 2D params
                if g_eff.ndim == 2:
                    update = _newton_schulz_5(g_eff, steps=ns_steps)
                else:
                    # Fallback for non-2D (shouldn't happen if used correctly)
                    update = g_eff

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


def split_params_for_muon(model, lr_muon=0.02, lr_adamw=1e-4, wd=0.01):
    """
    Split model parameters into Muon (2D matrices) and AdamW (everything else).

    Returns:
        muon_params: list of 2D weight tensors for Muon
        adamw_params: list of other tensors for AdamW
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Muon: 2D weight matrices (linear layers)
        if param.ndim == 2:
            muon_params.append(param)
        else:
            # AdamW: embeddings (3D in some cases), layernorm (1D), biases
            adamw_params.append(param)

    return muon_params, adamw_params
