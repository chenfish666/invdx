"""1D density filtering & projection, numpy+autograd.

This is the authoritative reference implementation: it runs unchanged inside
the meep conda env and anchors the parity tests for filters_jax. Never import
jax here.

Mapping chain (the standard filter-then-project pipeline):
    x  --(conic filter, radius R)-->  x_tilde  --(tanh projection, beta, eta)-->  rho
The filter is implemented as a dense matrix multiply W @ x, which autograd
differentiates exactly and cheaply for Nx ~ 1000.
"""

import numpy as np
import autograd.numpy as npa


def conic_filter_matrix(Nx, radius_um, grid_per_um):
    """Return W (Nx x Nx) implementing normalized conic (cone-kernel) filtering.

    Edge handling: kernel is renormalized per-row (equivalent to mirror-free
    'renormalize at boundary'), which avoids artificially eroding the edges.
    """
    dx = 1.0 / grid_per_um
    r_pix = radius_um / dx
    idx = np.arange(Nx)
    D = np.abs(idx[None, :] - idx[:, None])            # pixel distance matrix
    W = np.maximum(0.0, 1.0 - D / r_pix)               # conic kernel
    W = W / W.sum(axis=1, keepdims=True)               # row-normalize
    return W


def conic_kernel_2d(radius_um, dx_um, dy_um):
    """Small radial conic stencil K[i, j] = max(0, 1 - ||r||/R) on the
    Euclidean pixel-center distance ||r|| = sqrt((i*dx)^2 + (j*dy)^2).

    The 2D twin of `conic_filter_matrix`'s kernel: the neighbourhood is the
    Euclidean ball ||r_i - r_j|| <= R, so this is a genuine 2D cone — NOT
    separable into two 1D passes. Returned as a dense (2*ri+1, 2*rj+1) stencil, always
    odd-sized and centered; taps at distance >= R carry exactly 0. With
    dx != dy the stencil is elliptical in index space (isotropic in um).

    Unlike the 1D dense-matrix route, no normalization is baked in here: the
    caller divides by a ones-convolution (`conic_filter_2d`), which reproduces
    the per-row renormalization of `conic_filter_matrix` at the boundary and
    equals division by K.sum() in the interior.
    """
    if radius_um <= 0 or dx_um <= 0 or dy_um <= 0:
        raise ValueError(f"radius/dx/dy must be positive, got "
                         f"{radius_um=}, {dx_um=}, {dy_um=}")
    ri = int(np.floor(radius_um / dx_um))
    rj = int(np.floor(radius_um / dy_um))
    ii = np.arange(-ri, ri + 1)[:, None] * dx_um
    jj = np.arange(-rj, rj + 1)[None, :] * dy_um
    return np.maximum(0.0, 1.0 - np.sqrt(ii ** 2 + jj ** 2) / radius_um)


def conv2d_same(x, K):
    """Zero-padded 'same' 2D convolution by direct tap summation.

    Deliberately dependency-free (no scipy/jax: this module must keep running
    unchanged inside the meep conda env). K must be odd-sized in both axes —
    which `conic_kernel_2d` guarantees; the cost is O(taps * pixels), fine for
    the small stencils a physical filter radius produces. K is symmetric under
    180-degree rotation here, so convolution == correlation.
    """
    x = np.asarray(x, dtype=float)
    K = np.asarray(K, dtype=float)
    if K.shape[0] % 2 != 1 or K.shape[1] % 2 != 1:
        raise ValueError(f"kernel must be odd-sized, got {K.shape}")
    ri, rj = K.shape[0] // 2, K.shape[1] // 2
    xp = np.pad(x, ((ri, ri), (rj, rj)))
    out = np.zeros_like(x)
    for a in range(K.shape[0]):
        for b in range(K.shape[1]):
            if K[a, b] != 0.0:
                out += K[a, b] * xp[a:a + x.shape[0], b:b + x.shape[1]]
    return out


def conic_filter_2d(x, radius_um, dx_um, dy_um):
    """Normalized 2D conic filtering of a (nx, ny) density — numpy reference.

    filtered = (K (*) x) / (K (*) ones): interior pixels see the plain
    K-normalized cone, boundary pixels are renormalized over the part of the
    kernel that overlaps the design window — the same 'renormalize at
    boundary, no edge erosion' behaviour the 1D `conic_filter_matrix` gets
    from its per-row normalization. Constants are preserved exactly
    (filter(c * ones) == c * ones), which the unit tests pin.
    """
    K = conic_kernel_2d(radius_um, dx_um, dy_um)
    norm = conv2d_same(np.ones_like(np.asarray(x, dtype=float)), K)
    return conv2d_same(x, K) / norm


def tanh_projection(x, beta, eta):
    """Standard smoothed Heaviside projection (autograd-friendly)."""
    num = npa.tanh(beta * eta) + npa.tanh(beta * (x - eta))
    den = npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta))
    return num / den


def make_mapping(W):
    """Return mapping(x, eta, beta) -> projected density (autograd traceable)."""

    def mapping(x, eta, beta):
        xt = npa.dot(W, x)
        return tanh_projection(xt, beta, eta)

    return mapping


def softmin(values, beta_agg):
    """Smooth minimum via LogSumExp. `values` is an autograd array."""
    return -npa.log(npa.sum(npa.exp(-beta_agg * values))) / beta_agg


def softmin_weights(values_np, beta_agg):
    """Analytic d(softmin)/d(values) for combining pre-computed gradients."""
    v = np.asarray(values_np, dtype=float)
    e = np.exp(-beta_agg * (v - v.min()))              # shifted for stability
    return e / e.sum()


def is_traced(x):
    """True while autograd is tracing (values are ArrayBox, not ndarray)."""
    return type(x).__name__ == "ArrayBox"
