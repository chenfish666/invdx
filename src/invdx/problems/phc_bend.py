"""PhC 90-degree bend — standard benchmark in the photonic-crystal waveguide
literature:

    2D square lattice of dielectric rods, a = 1 um, rod radius R = 0.225a,
    eps_rod = 10 (n = 3.162), TM polarization (E along the rods — some
    engineering literature calls this "TE"; see the walkthrough), band gap
    at normalized frequency f = a/lambda = 0.29..0.41, in the spirit of
    Mekis et al.'s high-transmission PhC bend (Phys. Rev. Lett. 77, 3787,
    1996) and standard in PhC-waveguide textbooks (Joannopoulos, Johnson,
    Winn & Meade). A 90-degree bend waveguide is carved by omitting rods;
    point defects (extra omitted rods) near the bend perturb the
    transmission.

Role in invdx: showcase problem + regression benchmark connecting the
toolbox to a well-known literature reference. It runs on the two engines
that bracket the trust ladder — the self-written toy 2D FDTD (Layer C) and
Meep via the bridge (Layer A anchor) — and their transmission spectra are
compared as CURVES (conventions lesson 6), never at single frequencies.

Everything here is numpy-pure (no jax) so meep_worker can import this module
inside the meep env: the geometry is defined exactly once for both engines.

Units: lengths in lattice constants a, frequencies in c/a (= a/lambda).
With a = 1 um these are um and um/lambda respectively.
"""

from dataclasses import dataclass

import numpy as np

from ..config import BaseConfig
from .contract import ProblemSpec, Unsupported


@dataclass
class PhCBendConfig(BaseConfig):
    # ---- Lattice (literature values) ----
    r_rod: float = 0.225        # rod radius in units of a
    eps_rod: float = 10.0       # dielectric constant of the rods
    n_side: int = 21            # rods per side (odd: bend corner on a site)

    # ---- Band (reference gap f = 0.29..0.41) ----
    f_min: float = 0.20
    f_max: float = 0.50
    n_freq: int = 61

    # ---- Numerics (toy engine) ----
    res_per_a: int = 20         # grid cells per lattice constant
    pad_a: float = 1.0          # vacuum rim around the crystal (a units).
                                # Kept NARROW on purpose: a wide rim is a
                                # vacuum highway that lets light bypass the
                                # crystal and bury the band gap (measured:
                                # -2 dB instead of -20 dB with a 2.5a pad)
    bulk_cols: int = 8          # crystal thickness (periods) of the slab
                                # used for the band-gap measurement
    toy_steps: int = 6000
    toy_courant: float = 0.5

    # ---- Numerics (Meep side) ----
    meep_resolution: int = 20   # pixels per a

    @property
    def center(self):
        """Lattice index of the bend corner (input arm ends / output starts)."""
        return self.n_side // 2

    @property
    def freqs(self):
        return np.linspace(self.f_min, self.f_max, self.n_freq)


# --------------------------------------------------------------------------
# Geometry — defined once, consumed by both engines
# --------------------------------------------------------------------------


def rod_sites(n_side, layout, defect=None, bulk_cols=8):
    """Lattice sites (ix, iy) that KEEP their rod.

    The lattice carries one EXTRA ring of sites (indices -1 and n_side) so
    the crystal reaches into the boundary rim on all four sides — without it
    the rim is a vacuum highway around the crystal and the measured in-gap
    suppression is bypass-limited, not crystal-limited.

    layout — "bulk":     a bulk_cols-period-thick slab spanning the full
                         height (band-gap measurement at normal incidence)
             "straight": row iy = c removed edge to edge (reference
                         waveguide for the bend transmission ratio)
             "bend":     input arm (ix <= c, iy = c) + output arm
                         (ix = c, iy >= c) removed — light enters along +x
                         and leaves along +y
    defect — optional (di, dj): additionally remove the rod at
             (c + di, c + dj), the benchmark's point defect (site offsets
             are counted from the bend corner)
    """
    c = n_side // 2
    lo, hi = -1, n_side + 1                  # extra ring: -1 .. n_side
    if layout == "bulk":
        x0 = c - bulk_cols // 2
        return [(ix, iy) for ix in range(x0, x0 + bulk_cols)
                for iy in range(lo, hi)]
    removed = set()
    if layout == "straight":
        removed = {(ix, c) for ix in range(lo, hi)}
    elif layout == "bend":
        removed = {(ix, c) for ix in range(lo, c + 1)}
        removed |= {(c, iy) for iy in range(c, hi)}
    else:
        raise ValueError(f"unknown layout: {layout}")
    if defect is not None:
        di, dj = defect
        removed.add((c + di, c + dj))
    return [(ix, iy) for ix in range(lo, hi) for iy in range(lo, hi)
            if (ix, iy) not in removed]


def rod_centers_a(cfg, layout, defect=None):
    """Rod center coordinates in a-units, site (0,0) centered at
    (pad + 0.5, pad + 0.5); the extra ring sits inside the rim."""
    bulk_cols = getattr(cfg, "bulk_cols", 8)
    return [(cfg.pad_a + ix + 0.5, cfg.pad_a + iy + 0.5)
            for ix, iy in rod_sites(cfg.n_side, layout, defect, bulk_cols)]


def epsilon_grid(cfg, layout, defect=None):
    """Rasterized permittivity for the toy engine (binary fill; subpixel
    smoothing is a Meep luxury — expect small spectral shifts between the
    engines, which is precisely conventions lesson 6).

    layout "bulk_empty" is the rod-free vacuum grid (normalization run for
    the bulk band-gap measurement)."""
    n = int(round((cfg.n_side + 2 * cfg.pad_a) * cfg.res_per_a))
    eps = np.ones((n, n))
    if layout == "bulk_empty":
        return eps
    dx = 1.0 / cfg.res_per_a
    xs = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    r2 = cfg.r_rod ** 2
    for cx, cy in rod_centers_a(cfg, layout, defect):
        m = (X - cx) ** 2 + (Y - cy) ** 2 <= r2
        eps[m] = cfg.eps_rod
    return eps


# --------------------------------------------------------------------------
# Toy-engine transmission (Layer C)
# --------------------------------------------------------------------------


def _toy_ports(cfg):
    """Grid indices of sources and flux lines.

    Waveguide runs: point source ON the channel row, 1a inside the crystal
    entrance; the straight output line and the bend output line are both
    (n_side - 2)a of guided path away from it, so propagation effects cancel
    in the ratio.

    Bulk (band-gap) runs: a full-height line source in the vacuum in front
    of the slab and a full-height flux line behind it — normal-incidence
    transmission through bulk_cols periods.
    """
    res = cfg.res_per_a
    c = cfg.center
    n = int(round((cfg.n_side + 2 * cfg.pad_a) * res))
    row = int((cfg.pad_a + c + 0.5) * res)          # waveguide row/column
    i_src = int((cfg.pad_a + 1.0) * res)
    i_out = int((cfg.pad_a + cfg.n_side - 1.0) * res)
    half_w = int(1.5 * res)
    edge = max(2, res // 4)                          # keep off the Mur cells
    return {
        "src": {"i": i_src, "j": row},
        "bulk_src": {"i": i_src, "j0": edge, "j1": n - edge},
        # straight exit: vertical line, power flowing +x  (Sx = -Ez*Hy)
        "straight_out": ("x", i_out, row - half_w, row + half_w),
        # bend exit: horizontal line near the top, power flowing +y
        "bend_out": ("y", i_out, row - half_w, row + half_w),
        "bulk_out": ("x", i_out, edge, n - edge),
    }


def _toy_run(cfg, layout, defect, src_key, line_key):
    from ..toy import fdtd2d

    eps = epsilon_grid(cfg, layout, defect)
    ports = _toy_ports(cfg)
    n = eps.shape[0]
    dx = 1.0 / cfg.res_per_a
    # carrier-modulated pulse centered on the band: envelope width 1/(pi*hw)
    # puts the spectral amplitude at the band edges at e^-1 of the peak
    fcen = 0.5 * (cfg.f_min + cfg.f_max)
    spread = 1.0 / (np.pi * (cfg.f_max - cfg.f_min) / 2)
    out = fdtd2d.run(
        nx=n, ny=n, dx=dx, steps=cfg.toy_steps,
        source={**ports[src_key], "t0": 4 * spread, "spread": spread,
                "fcen": fcen},
        eps=eps, courant=cfg.toy_courant,
        line_probes={"out": ports[line_key]})
    dt = cfg.toy_courant * dx
    sign = -1.0 if ports[line_key][0] == "x" else +1.0
    return fdtd2d.line_flux_spectrum(out["lines"]["out"], cfg.freqs, dt, dx,
                                     sign=sign)


def toy_bend_transmission(cfg, defect=None):
    """T(f) of the 90-degree bend, normalized by the straight waveguide."""
    p_straight = _toy_run(cfg, "straight", None, "src", "straight_out")
    p_bend = _toy_run(cfg, "bend", defect, "src", "bend_out")
    T = p_bend / p_straight
    return {"freqs": cfg.freqs.tolist(), "T": T.tolist(),
            "P_straight": p_straight.tolist(), "P_bend": p_bend.tolist()}


def toy_bulk_transmission(cfg):
    """T(f) through a bulk_cols-period slab vs empty grid — locates the band
    gap. The reference gap f = 0.29..0.41 is the parameter-free anchor: if this
    curve's stopband lands elsewhere, the geometry or eps is wrong."""
    p_empty = _toy_run(cfg, "bulk_empty", None, "bulk_src", "bulk_out")
    p_bulk = _toy_run(cfg, "bulk", None, "bulk_src", "bulk_out")
    T = p_bulk / p_empty
    return {"freqs": cfg.freqs.tolist(), "T": T.tolist()}


def toy_field_map(cfg, layout="bend", defect=None, fstar=0.34):
    """Steady-state Ez(fstar) over the whole domain (the Meep-tutorial-style
    'light flowing through the structure' figure), from one pulsed run."""
    from ..toy import fdtd2d

    eps = epsilon_grid(cfg, layout, defect)
    ports = _toy_ports(cfg)
    fcen = 0.5 * (cfg.f_min + cfg.f_max)
    spread = 1.0 / (np.pi * (cfg.f_max - cfg.f_min) / 2)
    n = eps.shape[0]
    out = fdtd2d.run(
        nx=n, ny=n, dx=1.0 / cfg.res_per_a, steps=cfg.toy_steps,
        source={**ports["src"], "t0": 4 * spread, "spread": spread,
                "fcen": fcen},
        eps=eps, courant=cfg.toy_courant, field_dft_freqs=(fstar,))
    return {"field": out["field_dft"][fstar], "eps": eps,
            "fstar": float(fstar), "extent_a": (cfg.n_side + 2 * cfg.pad_a)}


# --------------------------------------------------------------------------
# Meep-engine transmission (Layer A anchor, via the bridge)
# --------------------------------------------------------------------------


def meep_payload(cfg, defect=None):
    """Plain-JSON payload for the meep_worker `phc_bend` task."""
    return {
        "n_side": cfg.n_side, "r_rod": cfg.r_rod, "eps_rod": cfg.eps_rod,
        "pad_a": cfg.pad_a, "resolution": cfg.meep_resolution,
        "f_min": cfg.f_min, "f_max": cfg.f_max, "n_freq": cfg.n_freq,
        "defect": list(defect) if defect else None,
        "dft_decay_tol": cfg.dft_decay_tol,
    }


def meep_bend_transmission(cfg, defect=None, n_ranks=1, timeout=1800):
    from ..engines import meep_bridge

    return meep_bridge.run_job("phc_bend", meep_payload(cfg, defect),
                               n_ranks=n_ranks, timeout=timeout)


# --------------------------------------------------------------------------
# Problem contract — the declaration the gates read (problems/contract.py)
# --------------------------------------------------------------------------

PROBLEM = ProblemSpec(
    name="phc_bend",
    config_cls=PhCBendConfig,
    gradcheck_case=Unsupported(
        "no differentiable design path — this module is numpy-pure on "
        "purpose so engines/meep_worker.py can import it inside the Meep "
        "environment, and the toy 2D FDTD it runs on has no adjoint. There "
        "is no gradient here for a finite difference to disagree with. "
        "Adding one means adding a jax path, at which point delete this "
        "declaration rather than loosening the gate."),
    reciprocity_case=Unsupported(
        "the measurement is p_bend / p_straight: two runs sharing one source, "
        "one detector family and one normalization, so the normalization "
        "cancels in the ratio and there is nothing left for G4 to check. The "
        "bend is of course reciprocal physically — what is missing is a "
        "mode-overlap normalization per port, which the toy engine does not "
        "have (no mode decomposition, only line flux). Wire this module to "
        "Meep's mode decomposition and G4 becomes applicable; change this "
        "declaration then."),
)
