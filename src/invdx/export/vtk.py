"""VTK export — turn run-dir field/permittivity snapshots into ParaView
files (.vti ImageData: our grids are uniform, so ImageData is the native
mapping — no mesh needed).

    python -m invdx.export.vtk runs/<run-dir>          # convert all known
    python -m invdx.export.vtk runs/<run-dir> -o viz/  # write elsewhere

Data-retention convention (the reason this works months later): every
measurement already snapshots its phasor fields as .npz in the run dir —
figures AND ParaView files are DERIVED artifacts, re-exportable from the
run dir alone. When you want full 3D volumes for ParaView (not just the
saved slices), run the measurement with its field/volume options on: disk
is cheap, re-simulation is not.

Exported point-data arrays per field file: Re, Im, Abs (|field|), and eps
(structure overlay) when present — so ParaView can contour the structure
and volume-render the field in one dataset.
"""

import argparse
import glob
import os

import numpy as np
from pyevtk.hl import imageToVTK


def _as3d(a):
    """pyevtk wants 3D arrays; pad 2D planes with a singleton z."""
    a = np.asarray(a)
    return a[:, :, None] if a.ndim == 2 else a


def export_field_npz(npz_path, out_base, spacing=(1.0, 1.0, 1.0),
                     origin=(0.0, 0.0, 0.0)):
    """One field_*.npz (field complex + eps [+ extent]) -> one .vti."""
    z = np.load(npz_path)
    field = np.asarray(z["field"])
    data = {
        "Re": np.ascontiguousarray(_as3d(np.real(field))),
        "Im": np.ascontiguousarray(_as3d(np.imag(field))),
        "Abs": np.ascontiguousarray(_as3d(np.abs(field))),
    }
    if "eps" in z:
        data["eps"] = np.ascontiguousarray(_as3d(np.asarray(z["eps"])))
    if "extent" in z:                       # (x0, x1, y0, y1) physical um
        x0, x1, y0, y1 = (float(v) for v in np.asarray(z["extent"]))
        nx, ny = field.shape[:2]
        # plain python floats: numpy scalars stringify as "np.float64(...)"
        # inside the XML Spacing attribute and break ParaView's parser
        spacing = (float((x1 - x0) / nx), float((y1 - y0) / ny), 1.0)
        origin = (x0, y0, 0.0)
    return imageToVTK(out_base, origin=origin, spacing=spacing,
                      pointData=data)


def export_eps_npy(npy_path, out_base, spacing=(1.0, 1.0, 1.0)):
    eps = np.ascontiguousarray(_as3d(np.load(npy_path)))
    return imageToVTK(out_base, spacing=spacing, pointData={"eps": eps})


def export_run(run_dir, out_dir=None):
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for fz in sorted(glob.glob(os.path.join(run_dir, "field_*.npz"))):
        base = os.path.join(out_dir, os.path.basename(fz)[:-4])
        made.append(export_field_npz(fz, base))
    for ep in sorted(glob.glob(os.path.join(run_dir, "eps_*.npy"))):
        base = os.path.join(out_dir, os.path.basename(ep)[:-4])
        made.append(export_eps_npy(ep, base))
    dz = os.path.join(run_dir, "design.npz")
    if os.path.exists(dz):
        z = np.load(dz)
        if "eps" in z:
            made.append(imageToVTK(
                os.path.join(out_dir, "design_eps"),
                pointData={"eps": np.ascontiguousarray(_as3d(z["eps"]))}))
    return made


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()
    made = export_run(args.run_dir, args.out)
    if not made:
        print(f"[vtk] nothing exportable in {args.run_dir}")
    for m in made:
        print(f"[vtk] {m}")


if __name__ == "__main__":
    main()
