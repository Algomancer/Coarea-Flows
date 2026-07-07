
"""
Coarea Normalizing Flow from Scratch -- Graph Parameterization
===============================================================

A minimal implementation of the coarea bijection as a normalizing-flow
layer, using a *graph-form* neural level function. Closed-form field and
divergence throughout: torch.compile-friendly.

Reference: arXiv:2605.08309 (coarea bijection, eqs. 0.5-0.9)
"""

import math

import torch
from torch import nn, Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# =============================================================================
# Device Configuration
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# =============================================================================
# The Coarea Layer
# =============================================================================
#
# The coarea bijection Phi(t, u) = eta(t, phi(u)) splits x into the value
# t = f(x) of a scalar level function and a coordinate u on the base level
# set f^{-1}(a), by flowing along
#
#     v(x) = grad f(x) / |grad f(x)|^2                          (0.5)
#
# so that f(eta(t, q)) = t exactly along trajectories            (0.6).
#
# We choose the *graph form*  f(x) = x_1 + g(x_rest)  with a single-hidden-
# layer g(x) = w2 . sigma(W1 x + b1) + b2. This buys three exact identities:
#
#   * global chart:      phi(u) = (a - g(u), u),  chart_inv(x) = x_rest
#   * no critical points: |grad f|^2 = 1 + |grad g|^2 >= 1
#   * base term is zero:  det[ v(phi(u)) | Dphi(u) ] = 1  (Schur)
#
# and, because g has one hidden layer, the divergence of v is closed form:
#
#     gam   = grad g = W1^T (sigma'(h) o w2)
#     s     = 1 + |gam|^2,      v = (1, gam) / s
#     tr H  = sum_j sigma''(h_j) w2_j ||row_j W1||^2     (exact Laplacian)
#     div v = tr H / s - 2 gam^T H gam / s^2
#
# The log-det-Jacobian is then purely the Liouville integral
#
#     log|det Phi'(t,u)| = int_a^t div v (eta(s, phi(u))) ds
#
# accumulated alongside the RK4 solve.
#

def gelu_d2(h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """GELU with first and second derivatives, closed form."""
    phi = torch.exp(-0.5 * h * h) * (1.0 / math.sqrt(2 * math.pi))
    Phi = 0.5 * (1.0 + torch.erf(h * (1.0 / math.sqrt(2))))
    return h * Phi, Phi + h * phi, (2.0 - h * h) * phi


class CoareaLayer(nn.Module):
    """Bijection x <-> z = (t, u) built from f(x) = x_1 + g(x_rest).

    forward:  x -> z = (t, u),  ldj = log|det dz/dx| = -log|det Phi'|
    inverse:  z -> x,           ldj = log|det dx/dz| = +log|det Phi'|
    """

    def __init__(self, dim: int, hidden: int = 64, a: float = 0.0, n_steps: int = 16):
        super().__init__()
        self.dim, self.a, self.n_steps = dim, a, n_steps
        self.W1 = nn.Parameter(torch.randn(hidden, dim - 1) / math.sqrt(dim - 1))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.zeros(hidden))   # zero-init -> identity layer
        self.b2 = nn.Parameter(torch.zeros(()))

    def g(self, xr: Tensor) -> Tensor:
        S, _, _ = gelu_d2(xr @ self.W1.T + self.b1)
        return S @ self.w2 + self.b2

    def field(self, x: Tensor, r1: Tensor) -> tuple[Tensor, Tensor]:
        """v(x) and its exact divergence; r1 = squared row norms of W1."""
        h = x[:, 1:] @ self.W1.T + self.b1
        _, Sp, Spp = gelu_d2(h)
        gam = (Sp * self.w2) @ self.W1
        s = 1.0 + (gam * gam).sum(-1, keepdim=True)
        v = torch.cat([1.0 / s, gam / s], dim=-1)
        c = Spp * self.w2
        q = gam @ self.W1.T
        se = s.squeeze(-1)
        div = (c * r1).sum(-1) / se - 2.0 * (c * q * q).sum(-1) / se ** 2
        return v, div

    def integrate(self, x: Tensor, t0: Tensor, t1: Tensor) -> tuple[Tensor, Tensor]:
        """RK4 for dx/ds = v(x) from per-sample time t0 to t1, jointly
        accumulating the Liouville integral of div v."""
        c = (t1 - t0).unsqueeze(-1)
        h = 1.0 / self.n_steps
        I = torch.zeros_like(t0)
        r1 = (self.W1 * self.W1).sum(-1)

        def F(xs: Tensor) -> tuple[Tensor, Tensor]:
            v, dv = self.field(xs, r1)
            return v * c, dv * c.squeeze(-1)

        for _ in range(self.n_steps):
            k1x, k1l = F(x)
            k2x, k2l = F(x + 0.5 * h * k1x)
            k3x, k3l = F(x + 0.5 * h * k2x)
            k4x, k4l = F(x + h * k3x)
            x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            I = I + (h / 6.0) * (k1l + 2 * k2l + 2 * k3l + k4l)
        return x, I

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        t = x[:, 0] + self.g(x[:, 1:])                
        x_a, I_down = self.integrate(x, t, torch.full_like(t, self.a))
        z = torch.cat([t.unsqueeze(-1), x_a[:, 1:]], dim=-1)
        return z, I_down

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        t, u = z[:, 0], z[:, 1:]
        q1 = self.a - self.g(u)                       # chart phi(u) = (a - g(u), u)
        q = torch.cat([q1.unsqueeze(-1), u], dim=-1)
        x, I_up = self.integrate(q, torch.full_like(t, self.a), t)
        return x, I_up

# =============================================================================
# Glue Layers and the Flow Stack
# =============================================================================
#
# One coarea layer levels a single direction. Stacking K of them behind
# fixed random rotations (ldj = 0) and per-dimension affine ActNorms lets
# each layer pick a fresh level direction; the base is a standard normal.
#

class ActNorm(nn.Module):
    """Per-dimension affine; initialize with data_init() before use."""

    def __init__(self, dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x * self.log_scale.exp() + self.shift, self.log_scale.sum().expand(x.shape[0])

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return (z - self.shift) * (-self.log_scale).exp(), (-self.log_scale.sum()).expand(z.shape[0])


class Rotation(nn.Module):
    """Fixed random orthogonal mixing; ldj = 0."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        Q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=torch.Generator().manual_seed(seed)))
        self.register_buffer("Q", Q)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x @ self.Q, x.new_zeros(x.shape[0])

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return z @ self.Q.T, z.new_zeros(z.shape[0])


class CoareaFlow(nn.Module):
    """[ActNorm -> Rotation -> CoareaLayer] x K -> ActNorm, N(0,I) base."""

    def __init__(self, dim: int, n_layers: int = 4, hidden: int = 64, n_steps: int = 16):
        super().__init__()
        self.dim = dim
        layers = []
        for k in range(n_layers):
            layers += [ActNorm(dim), Rotation(dim, seed=k), CoareaLayer(dim, hidden, n_steps=n_steps)]
        layers.append(ActNorm(dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ldj = x.new_zeros(x.shape[0])
        for layer in self.layers:
            x, d = layer(x)
            ldj = ldj + d
        return x, ldj

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        ldj = z.new_zeros(z.shape[0])
        for layer in reversed(self.layers):
            z, d = layer.inverse(z)
            ldj = ldj + d
        return z, ldj

    def log_prob(self, x: Tensor) -> Tensor:
        z, ldj = self.forward(x)
        return -0.5 * (z * z).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi) + ldj

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        x, _ = self.inverse(torch.randn(n, self.dim, device=DEVICE))
        return x


@torch.no_grad()
def data_init(model: CoareaFlow, x: Tensor):
    """One-shot data-dependent ActNorm init (eager, before compiling)."""
    for layer in model.layers:
        if isinstance(layer, ActNorm):
            std = x.std(0).clamp_min(1e-4)
            layer.log_scale.copy_(-std.log())
            layer.shift.copy_(-x.mean(0) / std)
        x, _ = layer(x)

# =============================================================================
# Data Generation
# =============================================================================

def gen_data(n: int, device: torch.device = DEVICE) -> Tensor:
    """Generate 2D mixture of 8 Gaussians arranged in a circle."""
    scale = 4.0
    centers = torch.tensor([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [-1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=torch.float32, device=device) * scale

    x = 0.5 * torch.randn(n, 2, device=device)
    center_ids = torch.randint(0, 8, (n,), device=device)
    x = (x + centers[center_ids]) / np.sqrt(2)
    return x

# =============================================================================
# Visualization
# =============================================================================

def viz_panels(model: CoareaFlow, filename: str):
    """Data / samples / density / first-layer level sets with flow paths."""
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    xd = gen_data(4000).cpu()
    ax[0].scatter(xd[:, 0], xd[:, 1], s=1, alpha=0.5)
    ax[0].set_title("data")

    xs = model.sample(4000).cpu()
    ax[1].scatter(xs[:, 0], xs[:, 1], s=1, alpha=0.5, color="C1")
    ax[1].set_title("coarea flow samples")

    g = torch.linspace(-4, 4, 200)
    XX, YY = torch.meshgrid(g, g, indexing="xy")
    P = torch.stack([XX.flatten(), YY.flatten()], -1).to(DEVICE)
    with torch.no_grad():
        L = model.log_prob(P).reshape(200, 200).cpu()
    L = L.clamp(min=L.max() - 12)          # log scale, 12-nat range: a single
    ax[2].imshow(L, origin="lower", extent=[-4, 4, -4, 4], cmap="magma")
    ax[2].set_title("model log-density")   # spike cannot black out the panel

    # level sets of the first coarea layer's f in its own input space,
    # with RK4 trajectories flowing data down to the base level f = a
    pre, layer = model.layers[:2], model.layers[2]
    x = gen_data(64)
    for m in pre:
        x, _ = m(x)
    with torch.no_grad():
        t = x[:, 0] + layer.g(x[:, 1:])
        path = [x.cpu()]
        xa = x
        for k in range(layer.n_steps):  # re-run integrate, recording steps
            frac0 = torch.full_like(t, layer.a)
            xa, _ = layer.integrate(x, t, t + (frac0 - t) * (k + 1) / layer.n_steps)
            path.append(xa.cpu())
        path = torch.stack(path)
        gl = torch.linspace(-3, 3, 120)
        GX, GY = torch.meshgrid(gl, gl, indexing="xy")
        G = torch.stack([GX.flatten(), GY.flatten()], -1).to(DEVICE)
        F = (G[:, 0] + layer.g(G[:, 1:])).reshape(120, 120).cpu()
    ax[3].contour(GX.cpu(), GY.cpu(), F, levels=15, cmap="coolwarm", linewidths=0.7)
    ax[3].contour(GX.cpu(), GY.cpu(), F, levels=[layer.a], colors="k", linewidths=2)
    ax[3].plot(path[:, :, 0], path[:, :, 1], color="green", alpha=0.4, lw=0.8)
    ax[3].set_title("layer-1 level sets of f + flow to f=a")

    for a in ax[:3]:
        a.set_xlim(-4, 4); a.set_ylim(-4, 4)
    plt.tight_layout()
    plt.savefig(filename, format="jpg", dpi=140, bbox_inches="tight")
    plt.close()

# =============================================================================
# Training
# =============================================================================

def train(model: CoareaFlow, n_iter: int = 6000, batch_size: int = 512, lr: float = 1e-3):
    """Maximum-likelihood training with a compiled loss."""
    data_init(model, gen_data(4096))
    loss_fn = torch.compile(lambda x: -model.log_prob(x).mean(), fullgraph=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = trange(n_iter)
    for i in pbar:
        loss = loss_fn(gen_data(batch_size))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        if (i + 1) % 100 == 0:
            pbar.set_description(f"nll: {loss.item():.4f}")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    model = CoareaFlow(dim=2, n_layers=4, hidden=64, n_steps=32).to(DEVICE)

    train(model)

    model.eval()
    viz_panels(model, filename="figs/toy_graph.jpg")

    # invertibility check
    with torch.no_grad():
        x = gen_data(1000)
        z, ldj_f = model(x)
        x_rec, ldj_i = model.inverse(z)
    print(f"roundtrip |x - x_rec|_max : {(x - x_rec).abs().max().item():.2e}")
    print(f"|ldj_f + ldj_i|_max       : {(ldj_f + ldj_i).abs().max().item():.2e}")
    print(f"final nll                 : {-model.log_prob(gen_data(4096)).mean().item():.4f}")
