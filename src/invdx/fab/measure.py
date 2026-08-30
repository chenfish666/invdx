"""Minimum-feature measurement & CD-corner edge bias.

Measurement only — never enters a differentiable path (numpy/scipy). fdtdx's
BrushConstraint2D can *enforce* feature sizes during optimization; these
utilities *verify* them on the final design, which is the independent check
a gate needs.
"""

import numpy as np


def min_feature_1d(rho, grid_per_um, thresh=0.5):
    """Return (min_solid_um, min_void_um) from run-length encoding of the
    binarized profile. Exact for 1D designs; use `imageruler` for 2D."""
    b = (np.asarray(rho) > thresh).astype(int)
    runs = {0: [], 1: []}
    val, count = b[0], 1
    for v in b[1:]:
        if v == val:
            count += 1
        else:
            runs[val].append(count)
            val, count = v, 1
    runs[val].append(count)
    dx = 1.0 / grid_per_um
    # Ignore boundary runs (waveguide connection / open end are not features).
    solid = runs[1][1:-1] if len(runs[1]) > 2 else runs[1]
    void = runs[0][1:-1] if len(runs[0]) > 2 else runs[0]
    ms = min(solid) * dx if solid else np.inf
    mv = min(void) * dx if void else np.inf
    return ms, mv


def erode_dilate_1d(rho_binary, delta_um, grid_per_um):
    """Uniform edge bias of +/-delta on a binary profile (CD-variation corner).

    delta_um > 0 : dilate solid (over-sized / under-etched)
    delta_um < 0 : erode solid  (under-sized / over-etched)
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    n = int(round(abs(delta_um) * grid_per_um))
    if n == 0:
        return rho_binary.copy()
    st = np.ones(2 * n + 1, dtype=bool)
    b = np.asarray(rho_binary) > 0.5
    out = binary_dilation(b, st) if delta_um > 0 else binary_erosion(b, st)
    return out.astype(float)
