"""Export a 1D design profile to GDS (gdstk), with a min-feature self-check.

Layout:
[access waveguide] -- [linear taper] -- [W-wide grating region]; the grating
teeth are straight lines extruded from the 1D profile. Layer (1, 0) = full
etch (map to the foundry layer table at tape-out time).

CLI:  python -m invdx.export.gds --design <run-dir-or-npy> --out out.gds
"""

import math

import numpy as np

from ..fab import measure


def export_profile_gds(rho, *, grid_per_um, width_um, min_feature_um, out,
                       taper_length_um=300.0, wg_length_um=20.0,
                       wg_width_um=0.45, cell_name="INVDX",
                       adiabatic_half_angle_deg=None):
    """Write the binary profile `rho` as grating teeth + taper + waveguide.

    Returns dict with the min-feature check, tooth count and the taper's half
    angle; raises nothing on rule violation — the caller decides (gates fail,
    humans may proceed).

    The taper's three numbers are conventional defaults on this signature, not
    derived ones: nothing here guarantees the taper is adiabatic, and a linear
    taper whose half angle only barely clears the applicable criterion carries
    an excess loss that no other check in this pipeline would notice. So the
    half angle is reported on every export: it is pure geometry, unambiguous,
    and it was always computable from arguments this function already had.

    adiabatic_half_angle_deg is the criterion to judge against. It defaults to
    None, and then no judgement is made — the admissible half angle depends on
    the waveguide stack and on the loss budget being designed to, so baking one
    number in here would turn a deliberate engineering choice made elsewhere
    into an anonymous constant.
    """
    import gdstk

    rho = (np.asarray(rho) > 0.5).astype(float)
    ms, mv = measure.min_feature_1d(rho, grid_per_um)
    rule_ok = min(ms, mv) >= min_feature_um - 1e-9

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(cell_name)
    dx = 1.0 / grid_per_um
    W = width_um
    x0 = 0.0  # grating starts at x = 0 in layout coordinates

    # grating teeth from run-length encoding
    b = rho > 0.5
    i = 0
    n_teeth = 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            cell.add(gdstk.rectangle((x0 + i * dx, -W / 2),
                                     (x0 + j * dx, W / 2), layer=1))
            n_teeth += 1
            i = j
        else:
            i += 1

    # taper + access waveguide (to the left of the grating)
    if taper_length_um > 0:
        cell.add(gdstk.Polygon([(-taper_length_um, -wg_width_um / 2),
                                (0, -W / 2), (0, W / 2),
                                (-taper_length_um, wg_width_um / 2)],
                               layer=1))
        xw = -taper_length_um
    else:
        xw = 0.0
    cell.add(gdstk.rectangle((xw - wg_length_um, -wg_width_um / 2),
                             (xw, wg_width_um / 2), layer=1))

    # Transitional rule while the geometry contract cannot see placement:
    # the converter on the other side silently discards cell-reference
    # offsets, collapsing every referenced shape onto the origin while total
    # area, polygon count and vertex count all stay identical. Until the
    # fingerprint carries layout extent on both sides of a real handoff, this
    # exporter emits one flat cell and asserts it.
    n_refs = sum(len(c.references) for c in lib.cells)
    if n_refs:
        raise ValueError(
            f"this exporter must emit a single flat cell, but the library "
            f"holds {n_refs} cell reference(s). The downstream converter drops "
            f"their placement offsets without changing area or any count, so "
            f"the handoff would be silently wrong.")
    lib.write_gds(out)

    half_angle = (math.degrees(math.atan(((W - wg_width_um) / 2)
                                         / taper_length_um))
                  if taper_length_um > 0 else None)
    margin = (adiabatic_half_angle_deg / half_angle
              if (half_angle and adiabatic_half_angle_deg) else None)
    return {"min_solid_um": float(ms), "min_void_um": float(mv),
            "rule_ok": bool(rule_ok), "n_teeth": int(n_teeth),
            "taper_half_angle_deg": half_angle,
            "taper_adiabatic_margin": margin,
            "out": str(out)}


def main():
    from ..cli import base_parser, apply_overrides, load_design
    from ..config import BaseConfig
    from . import contract

    p = base_parser(__doc__)
    p.add_argument("--design", required=True)
    p.add_argument("--out", default="design.gds")
    p.add_argument("--width", type=float, default=10.0,
                   help="grating region width W (um)")
    p.add_argument("--taper-length", type=float, default=300.0,
                   help="linear taper length (um); 0 disables the taper")
    p.add_argument("--wg-length", type=float, default=20.0)
    p.add_argument("--wg-width", type=float, default=0.45)
    p.add_argument("--allow-rule-violation", action="store_true",
                   help="write the file and exit 0 even if the minimum-feature "
                        "self-check fails")
    p.add_argument("--adiabatic-half-angle", type=float, default=None,
                   help="half-angle criterion (deg) to report the taper "
                        "margin against; omit to report the angle only")
    args = p.parse_args()
    cfg = apply_overrides(BaseConfig(), args)

    rho = load_design(args.design)
    res = export_profile_gds(
        rho, grid_per_um=cfg.design_grid_per_um, width_um=args.width,
        min_feature_um=cfg.min_feature, out=args.out,
        taper_length_um=args.taper_length, wg_length_um=args.wg_length,
        wg_width_um=args.wg_width,
        adiabatic_half_angle_deg=args.adiabatic_half_angle)
    print(f"[check] min solid {res['min_solid_um']*1e3:.0f} nm / "
          f"min void {res['min_void_um']*1e3:.0f} nm "
          f"(rule >= {cfg.min_feature*1e3:.0f} nm) -> "
          f"{'PASS' if res['rule_ok'] else 'FAIL'}")
    if res["taper_half_angle_deg"] is not None:
        print(f"[taper] half angle {res['taper_half_angle_deg']:.4f} deg "
              f"({args.wg_width} -> {args.width} um over "
              f"{args.taper_length} um)"
              + (f", margin {res['taper_adiabatic_margin']:.2f}x against the "
                 f"{args.adiabatic_half_angle} deg criterion"
                 if res["taper_adiabatic_margin"] else
                 "; no criterion given, so not judged"))
    # The geometry contract (contract.py) can only compare what came back
    # against a fingerprint taken at export time -- so take it now, or there
    # is nothing to compare when the converter's .ind file returns.
    fp_path = f"{res['out']}.fingerprint.json"
    contract.write_fingerprint(fp_path, contract.fingerprint_gds(res["out"]))
    print(f"[done] {res['out']}  ({res['n_teeth']} grating teeth)")
    print(f"[contract] polygon fingerprint written: {fp_path}")
    print("[note] inspect in KLayout; map layer (1,0) to the foundry "
          "full-etch layer before any tape-out.")
    # The library deliberately does not raise -- "the caller decides" -- but
    # nothing was deciding: this CLI printed FAIL and exited 0, and rule_ok
    # had no consumer anywhere. A rule violation that exits 0 is a rule that
    # is not enforced.
    if not res["rule_ok"] and not args.allow_rule_violation:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
