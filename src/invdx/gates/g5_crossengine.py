"""Gate 5 — cross-engine agreement: fdtdx (GPU) vs Meep (subprocess bridge)
on the same physical problem, plus the analytic answer as referee.

Problem: normal-incidence transmission of a lossless dielectric slab
(n=2, t=0.5um) at lambda=1.55um, each engine normalized by its own empty-cell
run — the ratio cancels power conventions *within* an engine; the explicit
convention contracts live in engines/conventions.py and matter as soon as
non-ratio quantities are compared.

Analytic (Airy, lossless slab in air):
    T = 1 / (1 + ((n^2-1)^2 / (4 n^2)) * sin^2(2 pi n t / lambda))

Skeleton tolerances are deliberately loose (10%): both engines run cheap,
under-resolved smoke settings here. Tighten together with resolution when a
concrete design problem lands.
"""

import numpy as np

from .runner import GateResult, NoProblem

NAME = "crossengine"
ORDER = 5
REQUIRES = ("gpu", "meep-env")
MEASURES_PROBLEM = NoProblem(
    "G5's device is its own: the dielectric slab hard-coded in the constants "
    "below, chosen because an analytic answer exists for it. The comparison "
    "is between two engines and that formula, so no problem module -- and no "
    "`--problem` -- has any claim on these transmissions")

N_SLAB = 2.0
T_SLAB_UM = 0.5
LAMBDA_UM = 1.55
TOL = 0.10
N_AVG = 108


def analytic_transmission():
    s = np.sin(2 * np.pi * N_SLAB * T_SLAB_UM / LAMBDA_UM) ** 2
    return 1.0 / (1.0 + ((N_SLAB ** 2 - 1) ** 2 / (4 * N_SLAB ** 2)) * s)


def _fdtdx_transmission(cfg):
    from invdx.engines import fdtdx_engine

    fluxes = {}
    for label, slab in (("empty", None), ("slab", (N_SLAB, T_SLAB_UM))):
        config, objs, cons, det = fdtdx_engine.vacuum_flux_scene(
            cfg, wavelength_um=LAMBDA_UM, periodic_sides=True, slab=slab)
        arrays = fdtdx_engine.run_forward(config, objs, cons, seed=cfg.seed)
        fluxes[label] = fdtdx_engine.steady_flux(arrays, det, n_avg=N_AVG)
    return fluxes["slab"] / fluxes["empty"]


def run(cfg, args):
    from invdx.engines import meep_bridge

    t_ref = analytic_transmission()
    t_fdtdx = _fdtdx_transmission(cfg)
    res = meep_bridge.run_job("slab_transmission", {
        "resolution": 20, "dft_decay_tol": cfg.dft_decay_tol,
        "n_slab": N_SLAB, "t_slab": T_SLAB_UM})
    t_meep = res["transmission"]

    details = {"T_analytic": float(t_ref), "T_fdtdx": float(t_fdtdx),
               "T_meep": float(t_meep),
               "fdtdx_vs_meep": float(abs(t_fdtdx - t_meep) / t_meep),
               "fdtdx_vs_analytic": float(abs(t_fdtdx - t_ref) / t_ref),
               "meep_vs_analytic": float(abs(t_meep - t_ref) / t_ref)}

    bad = [k for k in ("fdtdx_vs_meep", "fdtdx_vs_analytic",
                       "meep_vs_analytic") if details[k] > TOL]
    if bad:
        return GateResult(NAME, "fail", {
            "reason": f"cross-engine disagreement beyond {TOL:.0%}: {bad}",
            **details})
    return GateResult(NAME, "ok", details)
