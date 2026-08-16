"""Export a 1D design profile to GDS (gdstk), with a min-feature self-check.

Library port of pvgc/scripts/09_export_gds.py. Layout:
[access waveguide] -- [linear taper] -- [W-wide grating region]; the grating
teeth are straight lines extruded from the 1D profile. Layer (1, 0) = full
etch (map to the foundry layer table at tape-out time).

CLI:  python -m invdx.export.gds --design <run-dir-or-npy> --out out.gds
"""

import numpy as np

from ..fab import measure


def export_profile_gds(rho, *, grid_per_um, width_um, min_feature_um, out,
                       taper_length_um=300.0, wg_length_um=20.0,
                       wg_width_um=0.45, cell_name="INVDX"):
    """Write the binary profile `rho` as grating teeth + taper + waveguide.

    Returns dict with the min-feature check and tooth count; raises nothing
    on rule violation — the caller decides (gates fail, humans may proceed).
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

    lib.write_gds(out)
    return {"min_solid_um": float(ms), "min_void_um": float(mv),
            "rule_ok": bool(rule_ok), "n_teeth": int(n_teeth), "out": str(out)}


def main():
    from ..cli import base_parser, apply_overrides, load_design
    from ..config import BaseConfig

    p = base_parser(__doc__)
    p.add_argument("--design", required=True)
    p.add_argument("--out", default="design.gds")
    p.add_argument("--width", type=float, default=10.0,
                   help="grating region width W (um)")
    p.add_argument("--taper-length", type=float, default=300.0,
                   help="linear taper length (um); 0 disables the taper")
    p.add_argument("--wg-length", type=float, default=20.0)
    p.add_argument("--wg-width", type=float, default=0.45)
    args = p.parse_args()
    cfg = apply_overrides(BaseConfig(), args)

    rho = load_design(args.design)
    res = export_profile_gds(
        rho, grid_per_um=cfg.design_grid_per_um, width_um=args.width,
        min_feature_um=cfg.min_feature, out=args.out,
        taper_length_um=args.taper_length, wg_length_um=args.wg_length,
        wg_width_um=args.wg_width)
    print(f"[check] min solid {res['min_solid_um']*1e3:.0f} nm / "
          f"min void {res['min_void_um']*1e3:.0f} nm "
          f"(rule >= {cfg.min_feature*1e3:.0f} nm) -> "
          f"{'PASS' if res['rule_ok'] else 'FAIL'}")
    print(f"[done] {res['out']}  ({res['n_teeth']} grating teeth)")
    print("[note] inspect in KLayout; map layer (1,0) to the foundry "
          "full-etch layer before any tape-out.")


if __name__ == "__main__":
    main()
