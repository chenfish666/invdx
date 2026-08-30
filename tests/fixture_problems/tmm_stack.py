"""tmm_stack -- the smallest problem that can earn G2 Part C and G4.

A normal-incidence dielectric multilayer between two DIFFERENT half-infinite
media (air in, oxide out), solved by the characteristic-matrix method. No
FDTD, no engine, no GPU: transmittance is a closed-form product of 2x2
matrices, exact to machine precision and instant. That is the point -- it
makes the two gates' plumbing measurable without dragging a solver in.

Why this shape and not something simpler:

  * G4 needs two directions whose normalizations are derived INDEPENDENTLY,
    otherwise it passes vacuously (which is exactly why `phc_bend` declares
    it inapplicable). Here the two ambient admittances differ, so the
    amplitude transmissions are NOT equal in the two directions --
    t_LR != t_RL -- and only the properly normalized power transmittances
    are. Reciprocity of the transmittance is a theorem for this system, so
    the gate is checking a real invariant, and the classic slip of reporting
    |t|^2 as a transmittance breaks it by 10*log10(n_out/n_in) on one side
    only. `reciprocity_case(normalized=False)` reproduces that slip on
    purpose; tests/test_problem_contract.py asserts the gate catches it.

  * G2 Part C needs a differentiable design path with real gradient signal.
    The design vector is one latent per layer, run through the same
    `fab.filters_jax.tanh_projection` the production path uses, into the
    layer permittivities. jax differentiates the matrix product; the gate
    finite-differences it. A wrong chain rule here shows up the same way it
    would in a real problem.

This module is deliberately NOT in src/invdx/problems/ -- see
fixture_problems/__init__.py.
"""

from dataclasses import dataclass

import numpy as np

from invdx.config import BaseConfig
from invdx.problems import GradcheckCase, ProblemSpec, ReciprocityCase


@dataclass
class TMMStackConfig(BaseConfig):
    lam_um: float = 1.31            # design wavelength
    n_layers: int = 16              # design variables, one per layer
    d_um: float = 0.1323            # layer thickness, fixed (only the index
                                    # is designed); ~quarter-wave at the
                                    # mean of n_lo/n_hi
    n_lo: float = 1.447             # rho = 0  (oxide)
    n_hi: float = 3.503             # rho = 1  (silicon)
    n_in: float = 1.0               # incidence medium (air)
    n_out: float = 1.447            # exit medium (oxide). MUST differ from
                                    # n_in, or the two directions become
                                    # trivially identical and G4 checks
                                    # nothing here either.

    @property
    def indices_binary(self):
        """The starting stack: alternating hi/lo, as plain numpy."""
        alt = np.arange(self.n_layers) % 2
        return np.where(alt == 0, self.n_hi, self.n_lo)


# --------------------------------------------------------------------------
# Physics -- characteristic matrices, numpy side
# --------------------------------------------------------------------------


def _stack_amplitudes(indices, d_um, lam_um, y_out):
    """(B, C) of the characteristic-matrix product, terminated by `y_out`.

    M_j = [[cos d, i sin d / y_j], [i y_j sin d, cos d]],  d = 2 pi n_j t / lam
    [B, C] = (prod_j M_j) @ [1, y_out]
    Layers are multiplied in the order light meets them, so reversing the
    direction means reversing `indices` AND swapping the two ambients --
    both, or the "reverse" run is not the reverse of anything. The incidence
    admittance enters only in the caller's transmittance formula.
    """
    B, C = 1.0 + 0j, complex(y_out)
    for n in reversed(np.asarray(indices)):
        delta = 2.0 * np.pi * n * d_um / lam_um
        c, s = np.cos(delta), np.sin(delta)
        B, C = c * B + 1j * s / n * C, 1j * n * s * B + c * C
    return B, C


def transmission(cfg, indices, reverse=False, normalized=True):
    """Power transmittance through the stack, one direction.

    normalized=False returns |t|^2 instead of T -- the amplitude-squared
    mistaken for a transmittance. It is offered here, rather than only in a
    test, because the failure it produces is the one G4 exists to catch and a
    reader should be able to see both formulas side by side.
    """
    y_in, y_out = (cfg.n_out, cfg.n_in) if reverse else (cfg.n_in, cfg.n_out)
    idx = np.asarray(indices)[::-1] if reverse else np.asarray(indices)
    B, C = _stack_amplitudes(idx, cfg.d_um, cfg.lam_um, y_out)
    t2 = abs(2.0 * y_in / (y_in * B + C)) ** 2      # |t|^2, amplitude only
    if not normalized:
        return float(t2)
    return float((y_out / y_in) * t2)               # power crossing the exit


# --------------------------------------------------------------------------
# Differentiable design path -- jax side, built only when asked for
# --------------------------------------------------------------------------


def _jax_transmission_fn(cfg):
    """Return T(rho, beta) as a jax-traceable float32 function."""
    import jax.numpy as jnp

    from invdx.fab.filters_jax import tanh_projection

    eps_lo = cfg.n_lo ** 2
    eps_hi = cfg.n_hi ** 2
    y_in = jnp.asarray(cfg.n_in, dtype=jnp.complex64)
    y_out = jnp.asarray(cfg.n_out, dtype=jnp.complex64)

    def T(rho, beta):
        dens = tanh_projection(rho, beta, cfg.eta_i)
        n = jnp.sqrt(eps_lo + dens * (eps_hi - eps_lo))
        B = jnp.asarray(1.0, dtype=jnp.complex64)
        C = y_out
        for j in range(cfg.n_layers - 1, -1, -1):
            nj = n[j].astype(jnp.complex64)
            delta = 2.0 * jnp.pi * nj * cfg.d_um / cfg.lam_um
            c, s = jnp.cos(delta), jnp.sin(delta)
            B, C = c * B + 1j * s / nj * C, 1j * nj * s * B + c * C
        t2 = jnp.abs(2.0 * y_in / (y_in * B + C)) ** 2
        return jnp.real(y_out / y_in) * t2

    return T


# --------------------------------------------------------------------------
# Problem contract
# --------------------------------------------------------------------------


def gradcheck_case():
    """The alternating stack softened to 0.1/0.9, differentiated at beta[0].

    Same softening rule as `grating_coupler.gradcheck_case` and for the same
    reason: a uniform mid-grey stack is not a Bragg mirror and carries almost
    no signal, and a hard 0/1 latent sits on the clip boundary where a
    central difference silently becomes one-sided.
    """
    import jax
    import jax.numpy as jnp

    cfg = TMMStackConfig()
    T = _jax_transmission_fn(cfg)

    def loss(rho, beta):
        return -T(rho, beta)

    vg = jax.jit(jax.value_and_grad(loss))
    val = jax.jit(loss)

    binary = (cfg.indices_binary == cfg.n_hi).astype(float)
    base = 0.1 + 0.8 * binary
    beta = jnp.asarray(float(cfg.beta_schedule[0]), dtype=jnp.float32)

    return GradcheckCase(
        vg_fn=lambda p, b: vg(jnp.asarray(p, dtype=jnp.float32), b),
        value_fn=lambda p, b: float(val(jnp.asarray(p, dtype=jnp.float32), b)),
        base=base, beta=beta, seed=cfg.seed,
        info={"n_layers": cfg.n_layers, "lam_um": cfg.lam_um})


def reciprocity_case(normalized=True):
    """Forward (air -> stack -> oxide) vs reverse (oxide -> stack -> air).

    `normalized=False` drops the exit-side admittance ratio on BOTH sides;
    because the two ambients differ, that breaks the equality by
    10*log10(n_out/n_in) -- a one-sided error, which is the shape of bug this
    gate was written for. Default True, so the shipped declaration is the
    correct physics.
    """
    cfg = TMMStackConfig()
    idx = cfg.indices_binary
    fwd = transmission(cfg, idx, reverse=False, normalized=True)
    rev = transmission(cfg, idx, reverse=True, normalized=normalized)
    return ReciprocityCase(
        fwd_dB=float(10.0 * np.log10(fwd)),
        rev_dB=float(10.0 * np.log10(rev)),
        extra={"T_fwd": fwd, "T_rev": rev, "normalized_reverse": normalized})


PROBLEM = ProblemSpec(
    name="tmm_stack",
    config_cls=TMMStackConfig,
    gradcheck_case=gradcheck_case,
    reciprocity_case=reciprocity_case,
)
