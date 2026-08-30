"""Gate 3 — physics baseline: flux conservation in vacuum.

Periodic side boundaries give a clean 1D plane wave; the flux entering just
above the source must equal the flux leaving at the top of the cell. Cheap
skeleton version — concrete design problems add their own analytic anchors
(waveguide transmission, grating baselines) on top of this.
"""

from .runner import GateResult

NAME = "physics"
ORDER = 3
REQUIRES = ("gpu",)

TOL = 0.05
N_AVG = 108  # ~2 optical periods at lambda=1.55um, spacing 0.05um


def run(cfg, args):
    from invdx.engines import fdtdx_engine

    config, objs, cons, det = fdtdx_engine.vacuum_flux_scene(
        cfg, periodic_sides=True, with_input_detector=True)
    arrays = fdtdx_engine.run_forward(config, objs, cons, seed=cfg.seed)
    f_in = fdtdx_engine.steady_flux(arrays, "flux_in", n_avg=N_AVG)
    f_out = fdtdx_engine.steady_flux(arrays, "flux", n_avg=N_AVG)
    ratio = f_out / f_in
    details = {"flux_in": f_in, "flux_out": f_out, "ratio": ratio}
    if abs(ratio - 1.0) > TOL:
        return GateResult(NAME, "fail", {
            "reason": f"vacuum flux not conserved: out/in = {ratio:.4f} "
                      f"(|1-ratio| > {TOL})", **details})
    return GateResult(NAME, "ok", details)
