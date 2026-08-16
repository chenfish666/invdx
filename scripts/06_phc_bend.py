#!/usr/bin/env python
"""PhC 90-degree bend — the lab-lineage showcase problem, stage by stage.

Reproduces the professor's paper (square lattice a=1um, rods R=0.225a,
eps=10, TM band gap f=0.29..0.41; 90-degree bend carved by omitting rods;
point defects near the bend) on two engines: the self-written toy 2D FDTD
and Meep via the cross-env bridge.

Each stage is one hands-on step of docs/phc-bend-walkthrough.md:

  python scripts/06_phc_bend.py --stage eps      # look at the geometry
  python scripts/06_phc_bend.py --stage gap      # locate the band gap (toy)
  python scripts/06_phc_bend.py --stage bend     # bend transmission (toy)
  python scripts/06_phc_bend.py --stage meep     # Meep cross-check (bridge)
  python scripts/06_phc_bend.py --stage compare  # toy vs Meep, curve view
  python scripts/06_phc_bend.py --stage defect   # paper's point-defect sweep
  python scripts/06_phc_bend.py                  # eps + gap + bend

Every stage accepts --set (e.g. --set n_side=11 --set toy_steps=3000 for a
quick look, or --set eps_rod=8.9 to see the gap move).
"""

import json
import os

import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import phc_bend
from invdx import runio

GAP_REF = (0.29, 0.41)      # paper: band gap in normalized frequency a/lam


def bar(db, lo=-45):
    return "#" * max(0, int(db - lo))


def stage_eps(cfg, d):
    """ASCII view of the three layouts + rod counts."""
    for layout in ("bulk", "straight", "bend"):
        eps = phc_bend.epsilon_grid(cfg, layout)
        n_rods = len(phc_bend.rod_sites(cfg.n_side, layout,
                                        bulk_cols=cfg.bulk_cols))
        # downsample to one character per lattice site
        res = cfg.res_per_a
        print(f"\n[{layout}] {n_rods} rods, grid {eps.shape[0]}^2 "
              f"({cfg.res_per_a} cells/a)")
        m = int(round(cfg.n_side + 2 * cfg.pad_a))
        for jy in range(m - 1, -1, -1):          # print +y up
            row = ""
            for jx in range(m):
                cell = eps[jx * res:(jx + 1) * res, jy * res:(jy + 1) * res]
                row += "o" if cell.max() > 1 else "."
            print("   " + row)
        np.save(os.path.join(d, f"eps_{layout}.npy"), eps)
    print(f"\n[eps] arrays saved to {d}/eps_*.npy "
          f"(load them, plt.imshow(eps.T, origin='lower'))")


def stage_gap(cfg, d):
    """Toy bulk transmission -> band-gap location vs the paper."""
    res = phc_bend.toy_bulk_transmission(cfg)
    f = np.array(res["freqs"])
    Tdb = 10 * np.log10(np.abs(res["T"]) + 1e-12)
    print("\n   f      T_bulk")
    for i in range(len(f)):
        print(f" {f[i]:.3f}  {Tdb[i]:7.1f} dB  {bar(Tdb[i])}")
    deep = f[Tdb < -20]
    if len(deep):
        print(f"\n[gap] stopband (T < -20 dB): f = {deep.min():.3f}"
              f"..{deep.max():.3f}   paper: {GAP_REF[0]}..{GAP_REF[1]}")
    else:
        print("\n[gap] NO stopband found — geometry or eps is wrong")
    runio.save_json(os.path.join(d, "gap.json"), res)


def stage_bend(cfg, d, defect=None):
    """Toy bend transmission; only the in-gap portion is physical."""
    res = phc_bend.toy_bend_transmission(cfg, defect=defect)
    f = np.array(res["freqs"])
    T = np.array(res["T"])
    gap = (f >= GAP_REF[0]) & (f <= GAP_REF[1])
    print("\n   f      T_bend   (in-gap only: outside the gap the crystal "
          "does not confine the wave and the ratio is noise)")
    for i in np.where(gap)[0]:
        print(f" {f[i]:.3f}  {T[i]:7.3f}  " + "=" * int(max(T[i], 0) * 30))
    i_pk = np.argmax(np.where(gap, T, -np.inf))
    print(f"\n[bend] in-gap peak T = {T[i_pk]:.3f} @ f = {f[i_pk]:.3f}"
          + (f"   defect={defect}" if defect else ""))
    runio.save_json(os.path.join(d, "bend.json"),
                    {**res, "defect": list(defect) if defect else None})
    return f, T, gap


def stage_meep(cfg, d):
    """Meep cross-check through the bridge (4 runs, ~minutes on 1 rank)."""
    res = phc_bend.meep_bend_transmission(cfg, n_ranks=1)
    runio.save_json(os.path.join(d, "meep.json"), res)
    f = np.array(res["freqs"])
    Tk = 10 * np.log10(np.abs(res["T_bulk"]) + 1e-12)
    deep = f[Tk < -20]
    if len(deep):
        print(f"[meep] stopband: f = {deep.min():.3f}..{deep.max():.3f}   "
              f"paper: {GAP_REF[0]}..{GAP_REF[1]}")
    print(f"[meep] spectra saved to {d}/meep.json")


def stage_compare(cfg, d):
    """toy vs Meep, side by side — CURVES, not single frequencies
    (conventions lesson 6: engines discretize differently, features shift)."""
    mpath = os.path.join(d, "meep.json")
    if not os.path.exists(mpath):
        stage_meep(cfg, d)
    with open(mpath) as fh:
        meep = json.load(fh)
    toy = phc_bend.toy_bend_transmission(cfg)
    f_t = np.array(toy["freqs"])
    f_m = np.array(meep["freqs"])
    T_t, T_m = np.array(toy["T"]), np.array(meep["T_bend"])
    print("\n   f      T_toy    T_meep   (in-gap)")
    for i, fv in enumerate(f_t):
        if not (GAP_REF[0] <= fv <= GAP_REF[1]):
            continue
        j = int(np.argmin(np.abs(f_m - fv)))
        print(f" {fv:.3f}  {T_t[i]:7.3f}  {T_m[j]:7.3f}")
    gap_t = (f_t >= GAP_REF[0]) & (f_t <= GAP_REF[1])
    gap_m = (f_m >= GAP_REF[0]) & (f_m <= GAP_REF[1])
    print(f"\n[compare] in-gap mean T: toy {T_t[gap_t].mean():.3f}, "
          f"meep {T_m[gap_m].mean():.3f}")


def stage_field(cfg, d, defect=None, fstar=0.34):
    """Steady-state field map at an in-gap frequency: the money shot of
    light turning the 90-degree corner."""
    res = phc_bend.toy_field_map(cfg, "bend", defect, fstar)
    res["title"] = f"90° 彎穩態場 Re Ez  (f = {fstar})"
    np.savez(os.path.join(d, "field_bend.npz"), **res)
    print(f"[field] Ez({fstar}) saved -> {d}/field_bend.npz")
    from invdx.viz import plots
    L = res["extent_a"]
    out = plots.plot_field(res["field"], res["eps"],
                           os.path.join(d, "field_bend.png"),
                           title=res["title"], extent=(0, L, 0, L))
    print(f"[field] {out}")


DEFECTS = {
    # ring of candidate point defects around the bend corner, named by the
    # paper's layers (I = nearest) and orientation relative to the corner
    "I-horizontal": (1, 0), "I-vertical": (0, -1), "I-slant": (1, -1),
    "II-horizontal": (2, 0), "II-vertical": (0, -2), "II-slant": (2, -2),
}


def stage_defect(cfg, d):
    """Point-defect sweep: does the toy reproduce the paper's conclusion
    (vertical/horizontal defects matter more than slanted ones)?"""
    f0, T0, gap = stage_bend(cfg, d, defect=None)
    base = T0[gap].mean()
    print(f"\n[defect] no defect: in-gap mean T = {base:.3f}")
    rows = []
    for name, dd in DEFECTS.items():
        res = phc_bend.toy_bend_transmission(cfg, defect=dd)
        T = np.array(res["T"])[gap]
        rows.append((name, dd, T.mean(), T.mean() - base))
        print(f"[defect] {name:14s} {str(dd):8s} in-gap mean T = "
              f"{T.mean():.3f}  (delta {T.mean() - base:+.3f})")
    runio.save_json(os.path.join(d, "defects.json"),
                    [{"name": n, "defect": list(dd), "T_mean": tm,
                      "delta": dl} for n, dd, tm, dl in rows])
    print("\n[defect] paper's claim to check: |delta| of horizontal/vertical"
          " > |delta| of slant (within each layer)")


def main():
    p = base_parser(__doc__)
    p.add_argument("--stage", default=None,
                   choices=("eps", "gap", "bend", "meep", "compare",
                            "defect", "field"))
    p.add_argument("--fstar", type=float, default=0.34,
                   help="frequency for the field-map stage")
    p.add_argument("--defect", default=None,
                   help="di,dj lattice offset from the bend corner")
    args = p.parse_args()
    cfg = apply_overrides(phc_bend.PhCBendConfig(), args)
    d = start_run(cfg, args, "phc-bend")

    defect = None
    if args.defect:
        defect = tuple(int(v) for v in args.defect.split(","))

    if args.stage == "eps":
        stage_eps(cfg, d)
    elif args.stage == "gap":
        stage_gap(cfg, d)
    elif args.stage == "bend":
        stage_bend(cfg, d, defect)
    elif args.stage == "meep":
        stage_meep(cfg, d)
    elif args.stage == "compare":
        stage_compare(cfg, d)
    elif args.stage == "defect":
        stage_defect(cfg, d)
    elif args.stage == "field":
        stage_field(cfg, d, defect, args.fstar)
    else:
        stage_eps(cfg, d)
        stage_gap(cfg, d)
        stage_bend(cfg, d, defect)
    print(f"[done] {d}")


if __name__ == "__main__":
    main()
