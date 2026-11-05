#!/usr/bin/env python
"""grating_coupler cross-engine acceptance baseline: fdtdx against a Meep
reference, both computed here from the same nominal device.

Uniform fully-etched grating P=0.575um duty=0.5 on 220nm SOI, fiber-side
excitation (Gaussian beam), CE into the -x slab TE0 mode, at two incidence
angles:

  theta=10deg : off-vertical incidence -- the coupled case. Meep (through the
                bridge, worker task `grating_coupler`) and fdtdx measure
                CE(lambda) on one wavelength list with one normalization
                (grating_coupler_np.ce_sample). Acceptance is on the spectral
                ridge, never on one wavelength (conventions lesson 6): the two
                ridge peaks must agree within PEAK_TOL_DB; the ridge
                wavelengths are reported next to the effective-geometry
                differences that move them
  theta=0deg  : a symmetric grating at vertical incidence is physics-limited
                by second-order Bragg, so CE collapses -- an fdtdx-only
                expectation, not a cross-engine comparison

The Meep reference runs first -- it is the slow half, tens of minutes on a
few ranks -- and is saved as meep.json in the run directory before fdtdx
starts, so a later run can reuse it with --meep-json.

  python scripts/03_grating_coupler_baseline.py              # Meep, fdtdx, compare
  python scripts/03_grating_coupler_baseline.py --meep-only  # Meep reference alone (no jax)
  python scripts/03_grating_coupler_baseline.py --meep-json runs/<dir>/meep.json
  python scripts/03_grating_coupler_baseline.py --set spacing_um=0.025 --resolution 40

The last line is a quick look with both engines on a 25 nm grid. Exit status
is 1 when the ridge comparison fails.
"""

import json
import os
import sys
import time

import numpy as np

from invdx import runio
from invdx.cli import apply_overrides, base_parser, start_run
from invdx.problems import grating_coupler_np as gc

# Acceptance on the ridge peak (conventions lesson 6). The ridge wavelength is
# reported, not gated: it moves with the effective tooth width and slab
# thickness each engine builds from the same nominal device
# (effective_geometry below), which no single tolerance can absorb honestly.
PEAK_TOL_DB = 2.0

# What this script used to compare against: one number from a Meep run made
# outside the repository, with no script, settings or spectrum recorded.
# Kept so a reader can see what the old acceptance meant and whether the
# regenerated reference reproduces it; nothing here tests against it.
HISTORICAL_REFERENCE = {"CE_dB": -10.1, "lam_um": 1.310,
                        "source": "Meep run made outside the repository; "
                                  "settings not recorded"}


def parse_lams(spec):
    lo, hi, n = spec.split(",")
    return [float(v) for v in np.linspace(float(lo), float(hi), int(n))]


def effective_geometry(cfg, teeth):
    """What the two engines build from the nominal device, as far as that can
    be stated without running either.

    fdtdx fills whole cells: a block becomes round(size / dx) cells starting
    at cell round(edge / dx) (fdtdx's placement rule, which
    grating_coupler.rasterize_teeth mirrors). Meep subpixel-averages the
    nominal edges. Even where both sit on the same grid line, the two Yee
    grids can sample an edge up to half a cell apart -- the figure lesson 6
    quotes.
    """
    dx = cfg.spacing_um

    def snap(v):
        return round(v / dx) * dx

    edge = max((abs(snap(cfg.X0 + x) - (cfg.X0 + x)) for x, _ in teeth),
               default=0.0)
    width = max((abs(snap(w) - w) for _, w in teeth), default=0.0)
    return {"fdtdx_spacing_um": dx,
            "nominal_t_si_um": cfg.t_si,
            "fdtdx_t_si_um": snap(cfg.t_si),
            "fdtdx_tooth_edge_snap_max_um": edge,
            "fdtdx_tooth_width_snap_max_um": width,
            "yee_half_cell_um": dx / 2,
            "meep_resolution_px_per_um": cfg.resolution,
            "meep_edges": "subpixel-averaged at the nominal positions"}


def _diff(a, b, path=""):
    """Paths at which two JSON-shaped values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b))
                for p in _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b))
                for p in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [path or "<whole payload>"]


def load_reference(path, payload):
    """A stored Meep result, accepted only for the job this run would send.

    The payload is the complete description of the Meep run
    (grating_coupler_np.meep_payload), so equal payloads mean the stored
    reference IS the run this one would make; anything else is a reference
    for some other device or grid.
    """
    with open(path) as f:
        meep = json.load(f)
    diffs = _diff(meep.get("payload"), payload)
    if diffs:
        shown = ", ".join(diffs[:8]) + (" ..." if len(diffs) > 8 else "")
        raise SystemExit(
            f"--meep-json {path}: computed for a different Meep job than this "
            f"run would send, so it is not this run's reference.\n"
            f"  differs at: {shown}\n"
            f"  Drop --meep-json to regenerate it.")
    return meep


def describe(r):
    s = f"{r['CE_dB']:.2f} dB @ {r['lam_um']:.4f} um"
    if r["resolved"]:
        return s + (f"  (best sample {r['sample_CE_dB']:.2f} dB @ "
                    f"{r['sample_lam_um']:.3f})")
    return s + f"  [{r['note']}]"


def main():
    p = base_parser(__doc__)
    p.add_argument("--period", type=float, default=0.575)
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--lams", default="1.26,1.36,21", metavar="LO,HI,N",
                   help="wavelength list (um) both engines are sampled on")
    p.add_argument("--meep-ranks", type=int, default=2,
                   help="MPI ranks for the Meep reference")
    p.add_argument("--meep-timeout", type=float, default=4 * 3600,
                   help="seconds before the bridge gives up on the Meep job")
    p.add_argument("--meep-only", action="store_true",
                   help="regenerate the Meep reference and stop: no fdtdx, "
                        "and no jax needed")
    p.add_argument("--meep-json", default=None, metavar="PATH",
                   help="use this stored Meep reference instead of running "
                        "one; refused unless it was computed for the "
                        "payload this run would send")
    args = p.parse_args()
    if args.meep_only and args.meep_json:
        p.error("--meep-only with --meep-json would neither run nor compare")
    cfg = apply_overrides(gc.GratingCouplerConfig(), args)
    d = start_run(cfg, args, "coupler-baseline")

    teeth = gc.uniform_grating_teeth(cfg, period=args.period, duty=args.duty)
    lams = parse_lams(args.lams)
    geom = effective_geometry(cfg, teeth)
    results = {"historical_reference": HISTORICAL_REFERENCE,
               "effective_geometry": geom, "lams_um": lams}

    # theta=10, engine 1 of 2: the Meep reference
    cfg.theta_deg = 10.0
    payload = gc.meep_payload(cfg, teeth, lams)
    if args.meep_json:
        meep = load_reference(args.meep_json, payload)
        meep["reused_from"] = os.path.abspath(args.meep_json)
    else:
        t0 = time.time()
        meep = gc.meep_spectrum(cfg, teeth, lams, n_ranks=args.meep_ranks,
                                timeout=args.meep_timeout)
        meep["bridge_wall_s"] = round(time.time() - t0, 1)
    runio.save_json(os.path.join(d, "meep.json"), meep)
    r_meep = gc.ridge(meep["spectrum"])
    print(f"[meep]  theta=10 ridge: {describe(r_meep)}")
    print(f"[meep]  meep {meep['meep_version']}, res {meep['resolution']}, "
          f"{meep['processes']} ranks; lesson-1 check |flux/2P_in - 1| <= "
          f"{meep['checks']['flux_over_2P_in_max_dev']:.2e}")
    if args.meep_only:
        print(f"[done] {d}/meep.json")
        return

    # theta=10, engine 2 of 2: fdtdx
    from invdx.problems import grating_coupler

    t0 = time.time()
    spec = grating_coupler.characterize_spectrum(cfg, teeth, lams)
    spec["wall_s"] = round(time.time() - t0, 1)
    results["theta_10_spectrum"] = spec
    cmp = gc.compare_ridges(meep["spectrum"], spec["spectrum"], PEAK_TOL_DB)
    results["theta_10_ridge"] = cmp
    print(f"[fdtdx] theta=10 ridge: {describe(cmp['candidate'])}  "
          f"[{spec['wall_s']}s]")
    print(f"[compare] ridge peak, fdtdx - meep: {cmp['delta_peak_dB']:+.2f} "
          f"dB (tolerance {PEAK_TOL_DB} dB)  "
          f"[{'ok' if cmp['pass'] else 'FAIL'}]")
    print(f"[compare] ridge wavelength, fdtdx - meep: "
          f"{cmp['delta_lam_nm']:+.1f} nm (reported, not gated): fdtdx builds "
          f"t_si = {geom['fdtdx_t_si_um'] * 1e3:.1f} nm for the nominal "
          f"{geom['nominal_t_si_um'] * 1e3:.1f}, snaps tooth edges by up to "
          f"{geom['fdtdx_tooth_edge_snap_max_um'] * 1e3:.2f} nm and widths by "
          f"{geom['fdtdx_tooth_width_snap_max_um'] * 1e3:.2f} nm, and the two "
          f"Yee grids can sample an edge "
          f"{geom['yee_half_cell_um'] * 1e3:.2f} nm apart (lesson 6)")
    if not cmp["pass"]:
        print(f"[compare] {cmp['reason']}")
    h = HISTORICAL_REFERENCE
    print(f"[history] the out-of-repo reference this replaces: {h['CE_dB']} dB"
          f" @ {h['lam_um']:.3f} um ({h['source']}; not used for acceptance)")

    # theta=0: single-wavelength suppression check (broadband-deep, robust)
    cfg.theta_deg = 0.0
    t0 = time.time()
    res = grating_coupler.characterize(cfg, teeth)
    res["wall_s"] = round(time.time() - t0, 1)
    results["theta_0"] = res
    print(f"[baseline] theta=0: CE = {res['CE_dB']:.2f} dB (expect < -20 dB, "
          f"symmetric vertical suppression)  [{res['wall_s']}s]")

    runio.save_json(os.path.join(d, "results.json"), results)
    print(f"[done] {d}/results.json")
    if not cmp["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
