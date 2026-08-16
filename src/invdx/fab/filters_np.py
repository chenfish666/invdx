"""1D density filtering & projection, numpy+autograd (ported from pvgc/core.py).

This is the authoritative reference implementation: it runs unchanged inside
the meep conda env and anchors the parity tests for filters_jax. Never import
jax here.

Mapping chain (proposal Eq. 6-8):
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
