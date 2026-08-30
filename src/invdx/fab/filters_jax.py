"""jnp twins of filters_np for the fdtdx differentiable path.

Same signatures and semantics as filters_np (the authoritative numpy+autograd
reference); bound together by tests/test_np_jax_parity.py. The filter matrix
W itself is built with numpy (static, trace-time constant) — only its
application is traced.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .filters_np import conic_filter_matrix  # numpy builder, shared verbatim
from .filters_np import conic_kernel_2d, conv2d_same  # 2D twin, same builder


def tanh_projection(x, beta, eta):
    """Standard smoothed Heaviside projection (jax-traceable)."""
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta))
    return num / den


def make_mapping(W):
    """Return mapping(x, eta, beta) -> projected density (jax-traceable)."""
    Wj = jnp.asarray(W)

    def mapping(x, eta, beta):
        xt = Wj @ x
        return tanh_projection(xt, beta, eta)

    return mapping


def conv2d_same_jnp(x, K):
    """Zero-padded 'same' 2D convolution, jax-traceable twin of
    `filters_np.conv2d_same`. K must be odd-sized (conic_kernel_2d
    guarantees it) and 180-degree symmetric, so lax's cross-correlation
    equals the convolution the numpy reference computes."""
    lhs = jnp.asarray(x)[None, None]                       # (1, 1, nx, ny)
    rhs = jnp.asarray(K, dtype=lhs.dtype)[None, None]      # (1, 1, kx, ky)
    out = jax.lax.conv_general_dilated(lhs, rhs, window_strides=(1, 1),
                                       padding="SAME")
    return out[0, 0]


def make_conic_filter_2d(shape, radius_um, dx_um, dy_um):
    """filter_fn(x) -> normalized 2D conic filtering of an (nx, ny) array.

    jnp twin of `filters_np.conic_filter_2d` for a FIXED design shape: the
    stencil and the boundary-renormalization map (ones-convolution) are both
    static trace-time constants built with numpy; only the single conv2d of
    the density itself is traced. Parity with the numpy reference is pinned
    by tests/test_2d_freeform.py.
    """
    K = conic_kernel_2d(radius_um, dx_um, dy_um)           # numpy, static
    norm = conv2d_same(np.ones(shape, dtype=float), K)     # numpy, static

    def filter_fn(x):
        return conv2d_same_jnp(x, K) / jnp.asarray(norm, dtype=x.dtype)

    return filter_fn


def softmin(values, beta_agg):
    """Smooth minimum via LogSumExp (numerically stable jax version)."""
    return -jax.nn.logsumexp(-beta_agg * jnp.asarray(values)) / beta_agg
