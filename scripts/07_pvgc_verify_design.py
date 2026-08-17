#!/usr/bin/env python
"""Independent-engine verification of a pvgc design (the reason invdx has a
second engine): load a pvgc run's binarized design profile, re-measure it on
fdtdx, and report what the paper needs — CE spectrum, ridge, linewidth, and
optionally the CD-corner sensitivity.

  python scripts/07_pvgc_verify_design.py --run /root/pvgc/runs/<dir> \
      [--set spacing_um=0.010] [--corners] [--tag verify]

Reads ONLY from the pvgc run dir (never writes there). The pvgc config in
that dir supplies theta/band/design-grid; every value can still be
overridden with --set.

Grid choice: default spacing is 10 nm = exactly the pvgc design grid, so
tooth edges land ON fdtdx cell boundaries and the effective-duty
discretization error (conventions lesson 6; ridge moves ~2.4 nm per nm of
linewidth) is zero by construction.

Cross-engine acceptance is SPECTRAL: compare the CE(lambda) ridge shape and
peak against pvgc's own binarize-validation numbers, never a single
wavelength.
"""

import json
import os

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.fab import measure
from invdx.problems import pvgc
from invdx import runio


def load_pvgc_design(run_dir, rho_name="design_rho.npy"):
    rho = np.load(os.path.join(run_dir, rho_name))
    with open(os.path.join(run_dir, "config.json")) as f:
        pcfg = json.load(f)
    return rho, pcfg


def split_dual(cfg, rho_a, rho_b):
    """pvgc dual-etch convention (sims.profile_geometry_dual): full-height
    silicon where A AND B; shallow-etched (t_si - t_shallow) where A AND
    NOT B; air where NOT A. Returns (teeth_full, teeth_shallow)."""
    full = rho_a * rho_b
    shallow = rho_a * (1.0 - rho_b)
    return pvgc.profile_teeth(cfg, full), pvgc.profile_teeth(cfg, shallow)


def verify(cfg, rho_binary, lams_um, corners=False, s11=False,
           rho2_binary=None):
    """Linewidth + CE spectrum (+ CD corners) of a binary design profile.
    With rho2_binary the design is dual-etch (layer A = rho_binary, layer
    B = rho2_binary, pvgc convention)."""
    solid, void = measure.min_feature_1d(rho_binary, cfg.design_grid_per_um)
    if rho2_binary is None:
        teeth = pvgc.profile_teeth(cfg, rho_binary)
        shallow = None
    else:
        teeth, shallow = split_dual(cfg, rho_binary, rho2_binary)
    out = {
        "linewidth": {"min_solid_um": solid, "min_void_um": void,
                      "n_teeth": len(teeth)},
    }
    if rho2_binary is not None:
        sB, vB = measure.min_feature_1d(rho2_binary, cfg.design_grid_per_um)
        out["linewidth_layer2"] = {"min_solid_um": sB, "min_void_um": vB,
                                   "n_shallow": len(shallow)}

    p_in, azimuth_sign, _ = pvgc.calibrated_beam(cfg)
    spec = pvgc.characterize_spectrum(cfg, teeth, lams_um,
                                      azimuth_sign=azimuth_sign,
                                      shallow_teeth=shallow)
    out["spectrum"] = spec["spectrum"]
    out["peak"] = max(spec["spectrum"], key=lambda r: r["CE_dB"])
    bw, lam_lo, lam_hi, bw_note = pvgc.bandwidth_3db(spec["spectrum"])
    out["bandwidth_3db"] = {"bw_nm": bw * 1e3, "lam_lo_um": lam_lo,
                            "lam_hi_um": lam_hi, "note": bw_note}

    if s11:
        # waveguide-side run: S11 (back-reflection) + forward CE, which
        # doubles as a reciprocity check against the fiber-side spectrum
        ws = pvgc.wg_side_characterize(cfg, teeth, shallow_teeth=shallow)
        near = min(out["spectrum"],
                   key=lambda r: abs(r["lam_um"] - cfg.lam_c))
        out["s11"] = {"S11_dB": ws["S11_dB"], "CE_fwd_dB": ws["CE_fwd_dB"],
                      "CE_rev_dB_at_lam_c": near["CE_dB"],
                      "reciprocity_mismatch_dB":
                          abs(ws["CE_fwd_dB"] - near["CE_dB"])}

    if corners:
        # CD-variation corners: +/-10 nm uniform edge bias
        out["corners"] = {}
        for name, delta in (("erode_10nm", -0.010), ("dilate_10nm", +0.010)):
            rho_c = measure.erode_dilate_1d(rho_binary, delta,
                                            cfg.design_grid_per_um)
            if rho2_binary is None:
                teeth_c, shallow_c = pvgc.profile_teeth(cfg, rho_c), None
            else:
                # CD bias hits both lithography layers
                rho2_c = measure.erode_dilate_1d(rho2_binary, delta,
                                                 cfg.design_grid_per_um)
                teeth_c, shallow_c = split_dual(cfg, rho_c, rho2_c)
            spec_c = pvgc.characterize_spectrum(cfg, teeth_c, lams_um,
                                                azimuth_sign=azimuth_sign,
                                                shallow_teeth=shallow_c)
            out["corners"][name] = {
                "spectrum": spec_c["spectrum"],
                "peak": max(spec_c["spectrum"], key=lambda r: r["CE_dB"]),
            }
    return out


def main():
    p = base_parser(__doc__)
    p.add_argument("--run", required=True, help="pvgc run dir (read-only)")
    p.add_argument("--rho2-name", default="auto",
                   help="layer-B file for dual-etch designs; 'auto' picks "
                        "design_rho2.npy when present, 'none' disables")
    p.add_argument("--rho-name", default="design_rho.npy",
                   help="design file inside the run dir (e.g. "
                        "design_rho_ruled.npy for the rule-surgery variant)")
    p.add_argument("--corners", action="store_true",
                   help="also measure +/-10nm CD corners")
    p.add_argument("--s11", action="store_true",
                   help="also run the waveguide-side excitation: S11 + "
                        "reciprocity check (one extra simulation)")
    p.add_argument("--field", action="store_true",
                   help="also capture the coupling-region steady-state "
                        "field map at lam_c (one extra simulation)")
    p.add_argument("--field-3d", action="store_true",
                   help="3D field slices (side + top view) from one full-3D "
                        "run at the 575/16 nm grid; the top view shows "
                        "lateral spreading (quasi-2D cannot). ~12 min GPU")
    p.add_argument("--width", type=float, default=10.0,
                   help="grating width W for the 3D field run (um)")
    p.add_argument("--lam-lo", type=float, default=1.27)
    p.add_argument("--lam-hi", type=float, default=1.35)
    p.add_argument("--n-lam", type=int, default=9)
    args = p.parse_args()

    rho, pcfg = load_pvgc_design(args.run, args.rho_name)
    rho_binary = (rho > 0.5).astype(float)
    rho2_binary = None
    if args.rho2_name != "none":
        p2 = os.path.join(args.run, "design_rho2.npy"
                          if args.rho2_name == "auto" else args.rho2_name)
        if os.path.exists(p2):
            rho2_binary = (np.load(p2) > 0.5).astype(float)
            print(f"[verify] dual-etch design: layer B from "
                  f"{os.path.basename(p2)} (t_etch2 = "
                  f"{pcfg.get('t_etch2', '?')} um)")

    cfg = pvgc.PVGCConfig(
        spacing_um=0.010,               # = design grid: zero snap error
        theta_deg=float(pcfg.get("theta_deg", 0.0)),
        sim_time_s=1.5e-12,
    )
    cfg.design_grid_per_um = int(pcfg.get("design_grid_per_um", 100))
    cfg.min_feature = float(pcfg.get("min_feature", cfg.min_feature))
    if pcfg.get("t_etch2"):
        cfg.t_shallow = float(pcfg["t_etch2"])
    cfg = apply_overrides(cfg, args)
    d = start_run(cfg, args, "pvgc-verify")

    lams = list(np.linspace(args.lam_lo, args.lam_hi, args.n_lam))
    res = verify(cfg, rho_binary, lams, corners=args.corners, s11=args.s11,
                 rho2_binary=rho2_binary)
    res["source_run"] = os.path.abspath(args.run)
    res["rho_name"] = args.rho_name

    lw = res["linewidth"]
    print(f"[verify] design: {lw['n_teeth']} teeth, min solid "
          f"{lw['min_solid_um'] * 1e3:.0f} nm, min void "
          f"{lw['min_void_um'] * 1e3:.0f} nm  "
          f"(rule: {cfg.min_feature * 1e3:.0f} nm)")
    for r in res["spectrum"]:
        print(f"[verify]   {r['lam_um']:.3f} um: {r['CE_dB']:7.2f} dB")
    pk = res["peak"]
    print(f"[verify] ridge peak: {pk['CE_dB']:.2f} dB @ {pk['lam_um']:.3f} um")
    bwr = res["bandwidth_3db"]
    print(f"[verify] 3dB bandwidth: {bwr['bw_nm']:.1f} nm "
          f"({bwr['lam_lo_um']:.3f}-{bwr['lam_hi_um']:.3f} um)"
          + (f"  [{bwr['note']}]" if bwr["note"] else ""))
    if args.s11:
        s = res["s11"]
        print(f"[verify] S11 = {s['S11_dB']:.2f} dB; reciprocity mismatch "
              f"{s['reciprocity_mismatch_dB']:.2f} dB")
    if args.corners:
        for name, c in res["corners"].items():
            cpk = c["peak"]
            print(f"[verify] corner {name}: peak {cpk['CE_dB']:.2f} dB @ "
                  f"{cpk['lam_um']:.3f} um")

    if args.field:
        if rho2_binary is None:
            teeth, shallow = pvgc.profile_teeth(cfg, rho_binary), None
        else:
            teeth, shallow = split_dual(cfg, rho_binary, rho2_binary)
        fm = pvgc.field_map(cfg, teeth, shallow_teeth=shallow)
        np.savez(os.path.join(d, "field_coupler.npz"), **fm)
        from invdx.viz import plots
        out = plots.plot_field(fm["field"], fm["eps"],
                               os.path.join(d, "field_coupler.png"),
                               title=str(fm["title"]),
                               extent=tuple(fm["extent"]),
                               xlabel="x (µm)", ylabel="y (µm)")
        print(f"[verify] field map: {out}")

    if args.field_3d:
        from invdx.viz import plots

        # dedicated 3D-safe numerics: 575/16 nm grid (62.7M cells fits one
        # card; 10 nm would be ~180M+ and OOM). Field VISUAL only — CE
        # numbers stay with the quasi-2D chain at exact design-grid spacing.
        cfg3 = pvgc.PVGCConfig(spacing_um=0.575 / 16, sim_time_s=0.8e-12,
                               theta_deg=cfg.theta_deg)
        cfg3.design_grid_per_um = cfg.design_grid_per_um
        teeth = pvgc.profile_teeth(cfg3, rho_binary)
        fm3 = pvgc.field_map_3d(cfg3, teeth, wg_width_um=args.width)
        for view, fm in fm3.items():
            np.savez(os.path.join(d, f"field_3d_{view}.npz"), **fm)
            out = plots.plot_field(
                fm["field"], fm["eps"],
                os.path.join(d, f"field_3d_{view}.png"),
                title=str(fm["title"]), extent=tuple(fm["extent"]),
                xlabel="x (µm)",
                ylabel="y (µm)" if view == "xy" else "z (µm)")
            print(f"[verify] 3D field ({view}): {out}")

    runio.save_json(os.path.join(d, "results.json"), res)
    print(f"[done] {d}/results.json")


if __name__ == "__main__":
    main()
