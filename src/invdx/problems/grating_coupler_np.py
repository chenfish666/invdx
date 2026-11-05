"""The numpy-pure half of the grating_coupler problem: the config, the device
as data, the analytic slab TE0 mode, the overlap and line-power primitives
CE is built from, the spectral-ridge readout, and the job description of the
Meep reference run.

It exists because `grating_coupler` imports jax and fdtdx at the top, so
nothing defined there can be reached from inside the Meep environment, where
`engines/meep_worker.py` runs and neither package is installed. This module
imports numpy and `invdx.config` and nothing else. That is what lets the
worker's `grating_coupler` task build its device from the same `chip_blocks`
and `ports` the fdtdx scene is built from, launch the beam
`fiber_beam_profile` describes, and turn its fields into CE with the same
`ce_sample` the fdtdx chain calls: the geometry and the normalization are
each written once, for both engines, as `phc_bend`'s geometry is.

`grating_coupler` re-exports every public name defined here, so
`grating_coupler.X` keeps working everywhere it was used before the split.
Code that has to run in the Meep environment imports this module instead.

Coordinates are grating_coupler's throughout: x along the propagation
direction, y vertical with y = 0 at the bottom of the Si layer. The fdtdx
scene maps y onto its z axis; the Meep task uses these coordinates as they
are, shifted only by the cell centre.

Rule for this file: numpy and `invdx.config`, nothing else.
tests/test_grating_coupler_np.py imports it with jax, fdtdx, autograd and
meep made unimportable, so a stray import fails a test on any machine rather
than a Meep job on the one machine that has Meep.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from ..config import BaseConfig


@dataclass
class GratingCouplerConfig(BaseConfig):
    # ---- Wavelength (um) ----
    lam_c: float = 1.31

    # ---- Materials (constant index across the simulated band) ----
    n_si: float = 3.503
    n_sio2: float = 1.447

    # ---- Layer stack (um), grating_coupler y-coordinates: Si layer occupies [0, t_si] ----
    t_si: float = 0.220
    t_box: float = 3.0
    t_sub: float = 1.5

    # ---- Lateral geometry (um) ----
    L_design: float = 10.0
    L_design_y: float = 10.0        # y extent of the 2D free-form design
                                    # window (2D path only; the quasi-2D
                                    # 1D path never reads it). Default = the
                                    # full wg width; a narrower y-symmetric
                                    # window cuts variables without changing
                                    # what the method does.
    pad_x: float = 4.0
    air_above: float = 3.0
    dpml: float = 1.0

    # ---- Fiber / beam ----
    w0: float = 4.6                 # Gaussian waist radius = half the
                                    # mode-field diameter of a standard
                                    # single-mode fiber
    fiber_x0: float = 0.0
    fiber_line_y: float = 1.2       # monitor line above chip (grating_coupler y coords)
    src_beam_y: float = 2.2         # beam source plane (grating_coupler y coords)
    theta_deg: float = 0.0

    # ---- FOM (0.0 = legacy fiber-excited CE-only FOM) ----
    w_s11: float = 0.0              # FOM = CE - w_s11 * R11, R11 = linear
                                    # |S11|^2; MUST stay a float literal —
                                    # cli._cast_like casts --set overrides to
                                    # the default's type and int("0.3") raises

    # ---- Waveguide monitor / source (grating_coupler x coords) ----
    x_mon_wg: float = -6.5
    x_src_wg: float = -7.5
    wg_mon_height: float = 2.5
    wg_src_waist: float = 0.20      # z-waist of the wg-side excitation beam;
                                    # injection purity is irrelevant because
                                    # P_in is the measured FORWARD mode
                                    # overlap (radiation is filtered out),
                                    # i.e. the a_fwd normalization used
                                    # throughout this module

    # ---- Optional second lithography layer (70 nm shallow etch, a typical
    #      SOI process option; constraint-relaxation path) ----
    t_shallow: float = 0.070        # shallow-etch depth (um): etched regions
                                    # keep t_si - t_shallow of silicon

    # ---- fdtdx numerics ----
    spacing_um: float = 0.0125      # 12.5 nm cells = 80 cells per um
    n_y_cells: int = 4              # thin periodic axis; must be >1 because
                                    # the released GaussianPlaneSource squeezes
                                    # the transverse amplitude to 2D
    sim_time_s: float = 1.5e-12     # fixed run length (no adaptive stop in
                                    # released fdtdx); keep IDENTICAL between
                                    # a measurement run and its normalization
                                    # run so phasor scalings cancel

    # ---- Derived (coupler-coordinate -> scene-coordinate offsets, um) ----
    @property
    def X0(self):
        return self.L_design / 2 + self.pad_x + self.dpml

    @property
    def Z0(self):
        return self.t_box + self.t_sub + self.dpml

    @property
    def cell_x(self):
        return 2 * self.X0

    @property
    def cell_z(self):
        return self.Z0 + self.t_si + self.air_above + self.dpml


# --------------------------------------------------------------------------
# Analytic asymmetric-slab TE0 mode (exact in the 2D physics of this problem)
# --------------------------------------------------------------------------


def slab_te0_neff(lam_um, t_um, n_core, n_sub, n_cover):
    """Effective index of the TE0 mode of an asymmetric slab via bisection on
    the standard dispersion relation:
        kappa*t = atan(gamma_s/kappa) + atan(gamma_c/kappa)   (m = 0)
    with kappa = k0*sqrt(n_core^2-neff^2), gamma = k0*sqrt(neff^2-n_clad^2).
    """
    k0 = 2 * np.pi / lam_um

    def resid(neff):
        kap = k0 * np.sqrt(n_core ** 2 - neff ** 2)
        gs = k0 * np.sqrt(max(neff ** 2 - n_sub ** 2, 1e-15))
        gc = k0 * np.sqrt(max(neff ** 2 - n_cover ** 2, 1e-15))
        return kap * t_um - np.arctan(gs / kap) - np.arctan(gc / kap)

    lo, hi = max(n_sub, n_cover) + 1e-9, n_core - 1e-9
    if resid(lo) * resid(hi) > 0:
        raise ValueError("TE0 not guided for these parameters")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(lo) * resid(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def slab_te0_mode(zs_um, z_core_lo_um, cfg):
    """(Ey, Hz_fwd, n_eff) of the slab TE0 mode sampled at heights zs_um.

    Ey: cos inside the core, evanescent decay in SiO2 below / air above.
    Hz for the +x traveling mode is n_eff*Ey in fdtdx's eta0-normalized
    units; the -x traveler flips the sign of Hz.
    """
    lam, t = cfg.lam_c, cfg.t_si
    neff = slab_te0_neff(lam, t, cfg.n_si, cfg.n_sio2, 1.0)
    k0 = 2 * np.pi / lam
    kap = k0 * np.sqrt(cfg.n_si ** 2 - neff ** 2)
    gs = k0 * np.sqrt(neff ** 2 - cfg.n_sio2 ** 2)
    gc = k0 * np.sqrt(neff ** 2 - 1.0)
    # core field cos(kap*(z-z_lo) - phi_s) with tan(phi_s) = gamma_s/kappa
    phi_s = np.arctan2(gs, kap)

    z = np.asarray(zs_um) - z_core_lo_um
    core = np.cos(kap * z - phi_s)
    below = np.cos(-phi_s) * np.exp(gs * z)                     # z < 0
    above = np.cos(kap * t - phi_s) * np.exp(-gc * (z - t))     # z > t
    Ey = np.where(z < 0, below, np.where(z > t, above, core))
    return Ey, neff * Ey, neff


def overlap_power_directional(E, H, Em, Hm, dl):
    """Power carried by the (Em, Hm) mode in measured fields (E, H).

    a = 1/4 * sum(E*conj(Hm) + conj(Em)*H) dl ;  P = |a|^2 / Pm with
    Pm = 1/2 * Re sum(Em*conj(Hm)) dl. Physical 1/2-convention throughout —
    identical structure to invdx.modes.overlap_power but valid for modes
    where Hm != Em (slab mode: Hm = n_eff*Em). Direction is selected by the
    sign of Hm (backward mode: pass -Hm).
    """
    a = 0.25 * np.sum(E * np.conj(Hm) + np.conj(Em) * H) * dl
    # |Pm|: a backward mode carries negative signed power along +x, but the
    # coupled power magnitude is what CE needs
    Pm = abs(0.5 * np.real(np.sum(Em * np.conj(Hm))) * dl)
    return float(np.abs(a) ** 2 / Pm)


def phasor_line_power(E, H, dl):
    """|1/2 Re sum(E x H*)| through a line from phasor fields (per unit y)."""
    return float(abs(0.5 * np.real(np.sum(E * np.conj(H))) * dl))


def signed_poynting_flux_x(Ey, Hz, dl):
    """SIGNED real Poynting flux in +x through a line, from phasor (Ey, Hz).

    For this problem's polarization (E = Ey y-hat only; H = Hx x-hat +
    Hz z-hat, no Hy in the thin-periodic-y quasi-2D scene):
        S = 1/2 Re(E x H*) = (1/2 Re(Ey Hz*)) x-hat - (1/2 Re(Ey Hx*)) z-hat
    so Sx = 1/2 Re(Ey * conj(Hz)).

    energy_budget()'s judgment #2 in one line: `phasor_line_power` above
    takes abs() of exactly this quantity, which is correct when the number
    feeds a NORMALIZATION ratio (CE, mode power — direction is irrelevant,
    only magnitude is) but wrong the moment the number has to go into an
    energy-conservation SUM: a box's four face fluxes only cancel to zero
    if inflow and outflow keep opposite signs. Do not route energy-budget
    quantities through phasor_line_power's abs().
    """
    return float(0.5 * np.real(np.sum(Ey * np.conj(Hz))) * dl)


def signed_poynting_flux_z(Ey, Hx, dl):
    """SIGNED real Poynting flux in +z through a line, from phasor (Ey, Hx).

    Sz = -1/2 Re(Ey * conj(Hx)) — see `signed_poynting_flux_x` for the
    E x H* derivation (the minus sign is the y-hat x x-hat = -z-hat term).
    """
    return float(-0.5 * np.real(np.sum(Ey * np.conj(Hx))) * dl)


# --------------------------------------------------------------------------
# The device as data — defined once, consumed by both engines
# --------------------------------------------------------------------------


def uniform_grating_teeth(cfg, period, duty, n_periods=None):
    """[(x_min_um, width_um)] of Si teeth in grating_coupler x-coordinates."""
    x0 = -cfg.L_design / 2
    if n_periods is None:
        n_periods = int(cfg.L_design // period)
    return [(x0 + i * period, duty * period) for i in range(n_periods)]


def profile_teeth(cfg, rho_binary):
    """Run-length encode a binary design profile into teeth (grating_coupler x-coords)."""
    x0 = -cfg.L_design / 2
    dx = 1.0 / cfg.design_grid_per_um
    b = np.asarray(rho_binary) > 0.5
    teeth, i = [], 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            teeth.append((x0 + i * dx, (j - i) * dx))
            i = j
        else:
            i += 1
    return teeth


class Block(NamedTuple):
    """One material block of the chip, grating_coupler coordinates (um).

    `material` is a key ("si" or "sio2"), not an index: the index belongs to
    the config (`material_index`), so a `--set n_si=...` reaches both engines
    through the one field.
    """

    name: str
    material: str
    x_min: float
    x_size: float
    y_min: float
    y_size: float


def chip_blocks(cfg, teeth=None, shallow_teeth=None):
    """Every material block of the chip, in the order the fdtdx scene has
    always placed them: substrate, BOX, output waveguide slab, teeth, then
    the shallow-etch layer.

    `grating_coupler.build_scene` places exactly these, one
    UniformMaterialObject each, and the Meep reference (`meep_payload`) one
    mp.Block each, so the two engines are handed one device rather than two
    descriptions of it. No two blocks of different materials overlap, so the
    order decides nothing physical in either engine.

    teeth / shallow_teeth -- [(x_min_um, width_um)], see `build_scene`.
    """
    blocks = [
        # substrate Si extends through the bottom PML
        Block("substrate", "si", -cfg.X0, cfg.cell_x,
              -(cfg.t_box + cfg.t_sub + cfg.dpml), cfg.t_sub + cfg.dpml),
        Block("box", "sio2", -cfg.X0, cfg.cell_x, -cfg.t_box, cfg.t_box),
        # output waveguide slab: right edge at the design window start,
        # crossing the left PML
        Block("wg_slab", "si", -cfg.X0, cfg.X0 - cfg.L_design / 2,
              0.0, cfg.t_si),
    ]
    blocks += [Block(f"tooth{k}", "si", x_min, w, 0.0, cfg.t_si)
               for k, (x_min, w) in enumerate(teeth or [])]
    blocks += [Block(f"shallow{k}", "si", x_min, w, 0.0,
                     cfg.t_si - cfg.t_shallow)
               for k, (x_min, w) in enumerate(shallow_teeth or [])]
    return blocks


def material_index(cfg, material):
    """Refractive index of a `Block.material` key."""
    return {"si": cfg.n_si, "sio2": cfg.n_sio2}[material]


class Line(NamedTuple):
    """A source plane or monitor line, grating_coupler coordinates (um):
    centre (x, y) and extent (size_x, size_y), one of which is zero."""

    x: float
    y: float
    size_x: float
    size_y: float


def ports(cfg):
    """Where the fiber-side scene's beam source and its two CE monitors sit.

    beam       the fiber-side source plane (fdtdx object "beam")
    fiber_mon  incident-power line, 0.4 um below the fiber line: CE's
               denominator, read in the empty-cell run
    wg_mon     the x-plane where the -x TE0 power is read in the grating run

    `build_scene` places its fdtdx objects with the same arithmetic, and
    tests/test_grating_coupler_np.py places that scene and checks each object
    against these lines, so the Meep reference cannot drift onto other
    planes without a test noticing.
    """
    return {
        "beam": Line(cfg.fiber_x0, cfg.src_beam_y,
                     cfg.L_design + 2 * cfg.pad_x - 1, 0.0),
        "fiber_mon": Line(0.0, cfg.fiber_line_y - 0.4,
                          cfg.cell_x - 2 * cfg.dpml - 0.5, 0.0),
        "wg_mon": Line(cfg.x_mon_wg, cfg.t_si / 2, 0.0, cfg.wg_mon_height),
    }


def fiber_beam_profile(xs_um, x0, w0, theta_deg):
    """Amplitude and launch delay of the fiber-side beam along its source
    line, as the fdtdx scene injects it.

    fdtdx 0.6.2 builds this beam as a plane source tilted by
    `azimuth_angle = -theta` (`build_scene`). For a z-plane source with
    direction "-", `rotate_vector` turns the wave vector into
    (+sin theta, 0, -cos theta), and `calculate_time_offset_yee` delays each
    source cell by (r . k_hat)/c, so the cell at x launches at
    t - (x - x0) sin(theta)/c. Two consequences the Meep reference has to
    copy rather than approximate:

      - the tilt is a true time delay, so the beam leaves at theta at EVERY
        wavelength. A fixed spatial phase exp(i k0 sin(theta) x) would tilt
        it by arcsin(sin(theta) lam / lam_c) instead: +-0.4 deg across
        1.26-1.36 um at theta = 10;
      - positive theta points the beam's in-plane propagation toward +x,
        away from the output waveguide. Nothing measured on the fdtdx side
        can confirm that (`calibrated_beam` says why the sign of the phase
        slope there is not physical), so tests/test_grating_coupler_np.py
        pins it against fdtdx's own functions, and the Meep task checks it
        on its own field: the beam centroid must walk toward +x between the
        source and fiber_mon.

    The amplitude is the waist profile projected onto the tilted plane,
    exp(-((x - x0) cos(theta) / w0)^2), as fdtdx's source projection does.

    Returns (amplitude, delay), delay in units of length/c -- Meep's time
    unit when lengths are in um.
    """
    th = np.deg2rad(theta_deg)
    u = np.asarray(xs_um, dtype=float) - x0
    return np.exp(-((u * np.cos(th) / w0) ** 2)), u * np.sin(th)


# --------------------------------------------------------------------------
# CE(lambda) — the one normalization, for both engines
# --------------------------------------------------------------------------


def ce_sample(lam_um, E_fib, H_fib, E_wg, H_wg, zs_wg, dl, *, t_si, n_si,
              n_sio2):
    """One CE(lambda) sample, normalized the one way this problem does it.

    CE = P_mode / P_in, both in the physical 1/2 convention and both from
    phasor fields of two runs driven by the same source, so the source
    spectrum and the phasor scaling cancel at every wavelength:

      P_in    `phasor_line_power` of the incident beam on fiber_mon, from the
              EMPTY-cell run (an abs is taken, so the sign of H_fib is
              irrelevant)
      P_mode  `overlap_power_directional` against the -x traveling slab TE0
              mode on wg_mon, from the GRATING run. The mode is re-solved at
              lam_um (slab dispersion) for the nominal stack (t_si, n_si,
              n_sio2) and sampled at zs_wg (grating_coupler y, um)

    dl is the sample spacing of both lines. H_wg must be the H component
    whose product with E_wg gives +Sx, i.e. Sx = 1/2 Re(E_wg conj(H_wg)):
    fdtdx's Hz, Meep's -Hy (see engines/meep_worker.py).
    `grating_coupler.characterize_spectrum` and the Meep task both call this,
    which is what "normalized the way the fdtdx chain normalizes" means in
    code rather than in a comment.
    """
    p_in = phasor_line_power(E_fib, H_fib, dl)
    mode = SimpleNamespace(lam_c=lam_um, t_si=t_si, n_si=n_si, n_sio2=n_sio2)
    Em, Hm_fwd, neff = slab_te0_mode(zs_wg, 0.0, mode)
    # -x traveling mode: Hz flips sign
    p_mode = overlap_power_directional(E_wg, H_wg, Em, -Hm_fwd, dl)
    ce = p_mode / p_in
    return {"lam_um": float(lam_um), "CE": float(ce),
            "CE_dB": float(10 * np.log10(abs(ce) + 1e-15)),
            "P_in": p_in, "P_mode": p_mode, "n_eff": float(neff)}


# --------------------------------------------------------------------------
# The spectral ridge — what cross-engine acceptance is defined on
# --------------------------------------------------------------------------


def ridge(spectrum):
    """Where a CE(lambda) spectrum peaks, and how high (conventions lesson 6:
    engines are compared on this, never on one wavelength).

    `spectrum` is a list of rows with "lam_um" and "CE_dB", in any order.
    The highest sample is refined by the parabola through it and its two
    neighbours, in dB against lambda -- exact for a Gaussian-shaped ridge,
    whose logarithm is a parabola -- so the reported wavelength is not
    quantized to the sample spacing. The raw sample is reported beside it.

    A peak on the first or last sample is not a ridge inside the band: the
    maximum may lie outside the sampled range, so that case comes back with
    resolved=False and the raw sample, never an extrapolation.
    """
    rows = sorted(spectrum, key=lambda r: r["lam_um"])
    lam = np.array([r["lam_um"] for r in rows], dtype=float)
    db = np.array([r["CE_dB"] for r in rows], dtype=float)
    if lam.size < 3:
        raise ValueError(f"a ridge needs at least 3 samples, got {lam.size}")
    k = int(np.argmax(db))
    out = {"lam_um": float(lam[k]), "CE_dB": float(db[k]),
           "sample_lam_um": float(lam[k]), "sample_CE_dB": float(db[k]),
           "resolved": False, "note": None}
    if k in (0, lam.size - 1):
        side = "short" if k == 0 else "long"
        out["note"] = (f"peak on the {side}-wavelength edge of the sampled "
                       f"range: the ridge may lie outside it, so this is a "
                       f"bound, not a ridge -- widen the lambda span")
        return out
    h = lam[k + 1] - lam[k - 1]
    t = (lam[k - 1:k + 2] - lam[k]) / h
    a, b, c = np.polyfit(t, db[k - 1:k + 2], 2)
    out["resolved"] = True
    if a < 0:
        out["lam_um"] = float(lam[k] - b / (2 * a) * h)
        out["CE_dB"] = float(c - b * b / (4 * a))
    else:
        out["note"] = "flat top over three samples: the middle one is reported"
    return out


def compare_ridges(reference, candidate, peak_tol_db):
    """Cross-engine acceptance on the spectral ridge of two CE(lambda) lists.

    Passes when both ridges lie inside the sampled band and their peaks agree
    within peak_tol_db. The ridge-wavelength difference is reported and not
    gated: it moves with the effective geometry each engine discretizes the
    same nominal device into (lesson 6), which is a property of the two
    grids, not an error of either engine.
    """
    ref, cand = ridge(reference), ridge(candidate)
    out = {"reference": ref, "candidate": cand,
           "delta_peak_dB": cand["CE_dB"] - ref["CE_dB"],
           "delta_lam_nm": (cand["lam_um"] - ref["lam_um"]) * 1e3,
           "peak_tol_dB": float(peak_tol_db), "pass": True, "reason": None}
    unresolved = [name for name, r in (("reference", ref),
                                       ("candidate", cand))
                  if not r["resolved"]]
    if unresolved:
        out["pass"] = False
        out["reason"] = (f"no ridge inside the sampled band for "
                         f"{' and '.join(unresolved)}: a peak on the band "
                         f"edge is a bound, and two bounds are not a "
                         f"comparison -- widen the lambda span")
    elif abs(out["delta_peak_dB"]) > peak_tol_db:
        out["pass"] = False
        out["reason"] = (f"ridge peaks differ by {out['delta_peak_dB']:+.2f} "
                         f"dB, beyond the {peak_tol_db} dB tolerance")
    return out


# --------------------------------------------------------------------------
# Meep reference run (worker task `grating_coupler`, via the bridge)
# --------------------------------------------------------------------------

# Meep's time unit is length/c, here um/c (3.34 fs). The reference run stops
# when its DFT fields have decayed to cfg.dft_decay_tol (lesson 4: always
# passed explicitly); this cap only turns a run that never decays into an
# error instead of a hang. For scale: fdtdx's fixed run, sim_time_s = 1.5 ps,
# is ~450 um/c, and the default uniform grating's grating run, at res 80 with
# 101 wavelengths, decayed by t = 439 / 1116 / 4064 um/c at tolerance
# 1e-6 / 1e-8 / 1e-10.
MEEP_MAX_RUN_TIME = 20000.0


def meep_payload(cfg, teeth, lams_um):
    """The complete job description of the Meep reference run, plain JSON.

    Built on the parent side from the same `chip_blocks` and `ports` the
    fdtdx scene uses, so the payload IS the device Meep is handed: readable
    in the job directory's job.json, and comparable -- two equal payloads
    describe the same Meep run, which is how a stored reference is matched to
    a later comparison (scripts/03 --meep-json). It reads no fdtdx-only
    field (spacing_um, n_y_cells, sim_time_s, ...), so changing one of those
    leaves a stored reference valid.

    Coordinates are grating_coupler's; the worker shifts them by
    cell["center"] into Meep's cell-centred frame. The cell has the pixel
    counts fdtdx gives its volume, round(size * resolution), and fdtdx's
    lower-left corner (-X0, -Z0), so with resolution == 1/spacing_um the two
    grids share every grid line. Blocks that reach the cell edge are carried
    one PML thickness past it: fdtdx fills its cells up to the edge, while a
    Meep block ending exactly on the edge would be subpixel-averaged against
    the vacuum just outside it.
    """
    res = int(cfg.resolution)
    sx = round(cfg.cell_x * res) / res
    sy = round(cfg.cell_z * res) / res
    x_lo, y_lo = -cfg.X0, -cfg.Z0
    x_hi, y_hi = x_lo + sx, y_lo + sy
    eps = 0.5 / res
    blocks = []
    for b in chip_blocks(cfg, teeth):
        x0, x1 = b.x_min, b.x_min + b.x_size
        y0, y1 = b.y_min, b.y_min + b.y_size
        x0 = x0 - cfg.dpml if x0 <= x_lo + eps else x0
        x1 = x1 + cfg.dpml if x1 >= x_hi - eps else x1
        y0 = y0 - cfg.dpml if y0 <= y_lo + eps else y0
        y1 = y1 + cfg.dpml if y1 >= y_hi - eps else y1
        blocks.append({"name": b.name,
                       "index": float(material_index(cfg, b.material)),
                       "center": [(x0 + x1) / 2, (y0 + y1) / 2],
                       "size": [x1 - x0, y1 - y0]})
    p = ports(cfg)
    beam = p["beam"]
    fcen = 1.0 / cfg.lam_c
    return {
        "resolution": res,
        "cell": {"size": [sx, sy],
                 "center": [(x_lo + x_hi) / 2, (y_lo + y_hi) / 2],
                 "dpml": float(cfg.dpml)},
        "blocks": blocks,
        # the fdtdx pulse: centred on lam_c, spectral width 0.2 * c / lam_c
        "beam": {"x0": float(beam.x), "y": float(beam.y),
                 "width": float(beam.size_x), "pitch": 1.0 / res,
                 "w0": float(cfg.w0), "theta_deg": float(cfg.theta_deg),
                 "fcen": fcen, "fwidth": 0.2 * fcen},
        "monitors": {name: {"center": [float(ln.x), float(ln.y)],
                            "size": [float(ln.size_x), float(ln.size_y)]}
                     for name, ln in p.items() if name != "beam"},
        "lams_um": [float(lam) for lam in lams_um],
        "mode": {"t_si": float(cfg.t_si), "n_si": float(cfg.n_si),
                 "n_sio2": float(cfg.n_sio2)},
        "dft_decay_tol": float(cfg.dft_decay_tol),
        "max_run_time": MEEP_MAX_RUN_TIME,
    }


def meep_spectrum(cfg, teeth, lams_um, n_ranks=2, timeout=4 * 3600):
    """CE(lambda) of the device on Meep, through the bridge.

    Returns the worker's result: "spectrum" rows shaped like
    `characterize_spectrum`'s (plus the worker's own checks), and the
    "payload" it was run on.
    """
    from ..engines import meep_bridge

    return meep_bridge.run_job("grating_coupler",
                               meep_payload(cfg, teeth, lams_um),
                               n_ranks=n_ranks, timeout=timeout)
