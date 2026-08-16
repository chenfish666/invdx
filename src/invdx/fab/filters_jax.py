"""jnp twins of filters_np for the fdtdx differentiable path.

Same signatures and semantics as filters_np (the authoritative numpy+autograd
reference); bound together by tests/test_np_jax_parity.py. The filter matrix
W itself is built with numpy (static, trace-time constant) — only its
application is traced.
"""

import jax
import jax.numpy as jnp

from .filters_np import conic_filter_matrix  # numpy builder, shared verbatim


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


def softmin(values, beta_agg):
    """Smooth minimum via LogSumExp (numerically stable jax version)."""
    return -jax.nn.logsumexp(-beta_agg * jnp.asarray(values)) / beta_agg
