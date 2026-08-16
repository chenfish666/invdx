"""Gaussian fiber mode & overlap power (ported from pvgc/core.py).

POWER-CONVENTION CONTRACT (the 2x bug guard)
--------------------------------------------
These functions use the PHYSICAL half convention:  P = 1/2 Re(E x H*).
Meep's DFT fields, |alpha|^2 eigenmode coefficients and fluxes all OMIT the
1/2. Whenever an overlap computed here is normalized by a Meep-side power,
bridge the conventions explicitly:

    CE = 2 * overlap_power(Ez, Hx, Eg, Hg, Pg, dx) / p_in_meep

pvgc underestimated waveguide-side CE by exactly 2x until the reciprocity
gate exposed this (median mismatch 2.93 dB -> 0.150 dB after the fix).
See engines/conventions.py (meep_to_physical_power) — never bridge ad hoc.

The invariant that catches convention drift, enforced in tests/test_modes.py:
    overlap_power(Eg, Hg, Eg, Hg, Pg, dx) / Pg == 1  (exactly)
"""

import numpy as np
import autograd.numpy as npa


def gaussian_mode_1d(xs, x0, w0):
    """Target fiber mode sampled on the monitor line.

    Scalar paraxial approximation at the waist: Ez_g = exp(-((x-x0)/w0)^2),
    Hx_g = Ez_g (plane-wave relation in air, natural units eps0 = mu0 = 1).
    Returns (Eg, Hg, Pg, dx) with Pg = mode power = 0.5 * sum(|Eg|^2) dx.
    """
    xs = np.asarray(xs)
    dx = xs[1] - xs[0]
    Eg = np.exp(-((xs - x0) / w0) ** 2)
    Hg = Eg.copy()
    Pg = 0.5 * np.sum(np.abs(Eg) ** 2) * dx
    return Eg, Hg, Pg, dx


def overlap_power(Ez, Hx, Eg, Hg, Pg, dx):
    """Power coupled into the target mode (works on autograd arrays).

    a = 1/4 * integral( Ez*conj(Hg) + conj(Eg)*Hx ) dx ;  P = |a|^2 / Pg.
    Sanity: for (Ez, Hx) == (Eg, Hg) this returns exactly Pg, i.e. all the
    incident power — verified in tests/test_modes.py.
    """
    a = 0.25 * npa.sum(Ez * np.conj(Hg) + np.conj(Eg) * Hx) * dx
    return npa.abs(a) ** 2 / Pg
