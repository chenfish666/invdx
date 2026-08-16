"""PVGC problem — O-band perfectly-vertical grating coupler (iSiPP50G rules)
on the fdtdx engine, replicating /root/pvgc's Config B measurement:

    fiber-side Gaussian beam (angle theta) -> CE into the -x traveling slab
    TE0 mode, normalized by the incident beam power from an empty-cell run.

Coordinate mapping (pvgc 2D -> fdtdx quasi-2D):
    pvgc x (propagation) -> fdtdx x, offset +X0 so the cell starts at 0
    pvgc y (vertical)    -> fdtdx z, offset +Z0 (pvgc y=0 = Si layer bottom)
    out-of-plane         -> fdtdx y, single cell, periodic boundaries
    pvgc Ez polarization -> fdtdx E_y

Field-convention notes (fdtdx stores eta0-normalized H, so H carries E units):
    plane wave in air, -z propagation:  Hx =  Ey
    slab TE0 mode, +x propagation:      Hz =  n_eff * Ey  (-x: flip sign)
Both match the natural-unit relations pvgc/invdx.modes assume, so the same
overlap machinery applies. All CE building blocks (mode overlap numerator,
beam power denominator) are computed from PhasorDetector fields with the
same wavelength and the same run duration, so phasor scaling factors cancel
exactly — never mix phasor-derived and PoyntingFluxDetector-derived powers
in one ratio.

The slab TE0 mode is analytic (asymmetric slab: SiO2 below, air above),
which in the 2D physics of this problem is exact — no numerical mode solver
enters the measurement chain.
"""

from dataclasses import dataclass

import jax
import numpy as np

import fdtdx

from ..config import BaseConfig

UM = 1e-6


@dataclass
class PVGCConfig(BaseConfig):
    # ---- Wavelength (um) ----
    lam_c: float = 1.31

    # ---- Materials (constant index over O-band, as in pvgc) ----
    n_si: float = 3.503
    n_sio2: float = 1.447

    # ---- Layer stack (um), pvgc y-coordinates: Si layer occupies [0, t_si] ----
    t_si: float = 0.220
    t_box: float = 3.0
    t_sub: float = 1.5

    # ---- Lateral geometry (um) ----
    L_design: float = 10.0
    pad_x: float = 4.0
    air_above: float = 3.0
    dpml: float = 1.0

    # ---- Fiber / beam ----
    w0: float = 4.6                 # Gaussian waist radius (SMF-28 MFD/2)
    fiber_x0: float = 0.0
    fiber_line_y: float = 1.2       # monitor line above chip (pvgc y coords)
    src_beam_y: float = 2.2         # beam source plane (pvgc y coords)
    theta_deg: float = 0.0

    # ---- Waveguide monitor / source (pvgc x coords) ----
    x_mon_wg: float = -6.5
    x_src_wg: float = -7.5
    wg_mon_height: float = 2.5
    wg_src_waist: float = 0.20      # z-waist of the wg-side excitation beam;
                                    # injection purity is irrelevant because
                                    # P_in is the measured FORWARD mode
                                    # overlap (radiation is filtered out),
                                    # exactly like pvgc's a_fwd normalization

    # ---- Optional second lithography layer (iSiPP50G 70 nm shallow etch;
    #      constraint-relaxation path) ----
    t_shallow: float = 0.070        # shallow-etch depth (um): etched regions
                                    # keep t_si - t_shallow of silicon

    # ---- fdtdx numerics ----
    spacing_um: float = 0.0125      # 12.5 nm = pvgc res-80 equivalent
    n_y_cells: int = 4              # thin periodic axis; must be >1 because
                                    # the released GaussianPlaneSource squeezes
                                    # the transverse amplitude to 2D
    sim_time_s: float = 1.5e-12     # fixed run length (no adaptive stop in
                                    # released fdtdx); keep IDENTICAL between
                                    # a measurement run and its normalization
                                    # run so phasor scalings cancel

    # ---- Derived (pvgc-coordinate -> scene-coordinate offsets, um) ----
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


# --------------------------------------------------------------------------
# Scene construction (fdtdx, quasi-2D)
# --------------------------------------------------------------------------


def uniform_grating_teeth(cfg, period, duty, n_periods=None):
    """[(x_min_um, width_um)] of Si teeth in pvgc x-coordinates."""
    x0 = -cfg.L_design / 2
    if n_periods is None:
        n_periods = int(cfg.L_design // period)
    return [(x0 + i * period, duty * period) for i in range(n_periods)]


def profile_teeth(cfg, rho_binary):
    """Run-length encode a binary design profile into teeth (pvgc x-coords)."""
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


def _pml_cells(cfg):
    return int(round(cfg.dpml / cfg.spacing_um))


def build_scene(cfg, teeth=None, with_chip=True, azimuth_sign=1.0,
                excitation="fiber", with_field_map=False,
                shallow_teeth=None):
    """Object list + constraints for one run.

    teeth        — list of (x_min_um, width_um) Si teeth in pvgc coordinates
                   (None = no grating; wg slab still present when with_chip)
    shallow_teeth— optional second-layer regions: Si of height
                   t_si - t_shallow (the iSiPP50G 70 nm shallow etch leaves
                   150 nm). Layers are independent patterns; where a full
                   tooth and a shallow region overlap, full silicon wins
                   (blocks simply stack)
    with_chip    — False gives the pvgc `beam_input_power` empty air cell
    excitation   — "fiber": Gaussian beam from above (pvgc Config B)
                   "wg": +x beam launched inside the waveguide (Config A);
                   injection impurity is normalized away by the forward
                   mode overlap at wg_mon
    azimuth_sign — diagnostic override of the fiber-beam tilt direction;
                   production code leaves it at 1.0 (azimuth = -theta)

    Detectors always present:
      "wg_mon"    x-plane at cfg.x_mon_wg, components (Ey, Hz)
      "fiber_mon" z-plane 0.4 um below the fiber line, components (Ey, Hx)
    Returns (config, object_list, constraints).
    """
    import jax.numpy as jnp

    spacing = cfg.spacing_um * UM
    sim_config = fdtdx.SimulationConfig(
        time=cfg.sim_time_s,
        resolution=spacing,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    object_list, constraints = [], []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(cfg.cell_x * UM, cfg.n_y_cells * cfg.spacing_um * UM,
                            cfg.cell_z * UM))
    object_list.append(volume)

    npml = _pml_cells(cfg)
    bound_cfg = fdtdx.BoundaryConfig(
        boundary_type_minx="pml", boundary_type_maxx="pml",
        boundary_type_miny="periodic", boundary_type_maxy="periodic",
        boundary_type_minz="pml", boundary_type_maxz="pml",
        thickness_grid_minx=npml, thickness_grid_maxx=npml,
        thickness_grid_miny=1, thickness_grid_maxy=1,
        thickness_grid_minz=npml, thickness_grid_maxz=npml,
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(bound_dict.values())

    si = fdtdx.Material(permittivity=cfg.n_si ** 2)
    sio2 = fdtdx.Material(permittivity=cfg.n_sio2 ** 2)

    def block(name, material, x_min, x_size, z_min, z_size):
        """Si/SiO2 block in pvgc coordinates, spanning the full thin y axis."""
        obj = fdtdx.UniformMaterialObject(
            name=name, material=material,
            partial_real_shape=(x_size * UM, None, z_size * UM))
        constraints.append(obj.place_relative_to(
            volume, axes=(0, 2), own_positions=(-1, -1),
            other_positions=(-1, -1),
            margins=((cfg.X0 + x_min) * UM, (cfg.Z0 + z_min) * UM)))
        constraints.append(obj.same_size(volume, axes=(1,)))
        object_list.append(obj)

    if with_chip:
        # substrate Si extends through the bottom PML (pvgc _stack_blocks)
        block("substrate", si, -cfg.X0, cfg.cell_x,
              -(cfg.t_box + cfg.t_sub + cfg.dpml), cfg.t_sub + cfg.dpml)
        block("box", sio2, -cfg.X0, cfg.cell_x, -cfg.t_box, cfg.t_box)
        # output waveguide slab: right edge at the design window start,
        # crossing the left PML
        block("wg_slab", si, -cfg.X0, cfg.X0 - cfg.L_design / 2,
              0.0, cfg.t_si)
        for k, (x_min, w) in enumerate(teeth or []):
            block(f"tooth{k}", si, x_min, w, 0.0, cfg.t_si)
        for k, (x_min, w) in enumerate(shallow_teeth or []):
            block(f"shallow{k}", si, x_min, w, 0.0,
                  cfg.t_si - cfg.t_shallow)

    # ---- Source ----
    # invdx's GaussianBeamSource, not fdtdx.GaussianPlaneSource: the released
    # 0.6.2 profile builder NaNs on strongly rectangular planes (see
    # engines/fdtdx_fixes.py). Amplitude exp(-((x-x0)/w0)^2) matches pvgc.
    from ..engines.fdtdx_fixes import GaussianBeamSource

    if excitation == "wg":
        # Config A: +x beam inside the waveguide core (z-Gaussian profile)
        wg_src = GaussianBeamSource(
            name="wg_src",
            partial_grid_shape=(1, None, None),
            partial_real_shape=(None, None, cfg.wg_mon_height * UM),
            wave_character=fdtdx.WaveCharacter(wavelength=cfg.lam_c * UM),
            temporal_profile=fdtdx.GaussianPulseProfile(
                center_wave=fdtdx.WaveCharacter(wavelength=cfg.lam_c * UM),
                spectral_width=fdtdx.WaveCharacter(
                    frequency=0.2 * 2.998e14 / cfg.lam_c)),
            direction="+",
            fixed_E_polarization_vector=(0.0, 1.0, 0.0),
            waist_radius=cfg.wg_src_waist * UM,
            profile_axis="vertical",
        )
        z_src_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
        constraints.append(wg_src.place_relative_to(
            volume, axes=(0, 2), own_positions=(-1, -1),
            other_positions=(-1, -1),
            margins=((cfg.X0 + cfg.x_src_wg) * UM,
                     (cfg.Z0 + z_src_lo) * UM)))
        constraints.append(wg_src.same_size(volume, axes=(1,)))
        object_list.append(wg_src)

    if excitation == "fiber":
        src_width = cfg.L_design + 2 * cfg.pad_x - 1
        source = GaussianBeamSource(
            name="beam",
            partial_real_shape=(src_width * UM, None, None),
            partial_grid_shape=(None, None, 1),
            wave_character=fdtdx.WaveCharacter(wavelength=cfg.lam_c * UM),
            temporal_profile=fdtdx.GaussianPulseProfile(
                center_wave=fdtdx.WaveCharacter(wavelength=cfg.lam_c * UM),
                spectral_width=fdtdx.WaveCharacter(
                    frequency=0.2 * 2.998e14 / cfg.lam_c)),
            direction="-",
            # for a z-plane source horizontal_axis=x, and rotate_vector's
            # AZIMUTH tilts propagation toward the horizontal axis —
            # elevation would tilt into the thin periodic y axis.
            azimuth_angle=azimuth_sign * (-cfg.theta_deg),
            fixed_E_polarization_vector=(0.0, 1.0, 0.0),
            waist_radius=cfg.w0 * UM,
        )
        constraints.append(source.place_relative_to(
            volume, axes=(0, 2), own_positions=(-1, -1),
            other_positions=(-1, -1),
            margins=(((cfg.X0 + cfg.fiber_x0 - src_width / 2)) * UM,
                     (cfg.Z0 + cfg.src_beam_y) * UM)))
        constraints.append(source.same_size(volume, axes=(1,)))
        object_list.append(source)

    # multi-wavelength support: set cfg._lams_um for a dense spectrum from a
    # single run (the v2 sparse-sampling lesson made this a first-class need)
    lams_um = getattr(cfg, "_lams_um", None) or (cfg.lam_c,)
    wave_chars = tuple(fdtdx.WaveCharacter(wavelength=l * UM) for l in lams_um)

    wg_mon = fdtdx.PhasorDetector(
        name="wg_mon",
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, None, cfg.wg_mon_height * UM),
        wave_characters=wave_chars,
        components=("Ey", "Hz"),
        exact_interpolation=True,
    )
    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    constraints.append(wg_mon.place_relative_to(
        volume, axes=(0, 2), own_positions=(-1, -1), other_positions=(-1, -1),
        margins=((cfg.X0 + cfg.x_mon_wg) * UM, (cfg.Z0 + z_mon_lo) * UM)))
    constraints.append(wg_mon.same_size(volume, axes=(1,)))
    object_list.append(wg_mon)

    fib_width = cfg.cell_x - 2 * cfg.dpml - 0.5
    fiber_mon = fdtdx.PhasorDetector(
        name="fiber_mon",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(fib_width * UM, None, None),
        wave_characters=wave_chars,
        components=("Ey", "Hx"),
        exact_interpolation=True,
    )
    constraints.append(fiber_mon.place_relative_to(
        volume, axes=(0, 2), own_positions=(-1, -1), other_positions=(-1, -1),
        margins=(((cfg.cell_x - fib_width) / 2) * UM,
                 (cfg.Z0 + cfg.fiber_line_y - 0.4) * UM)))
    constraints.append(fiber_mon.same_size(volume, axes=(1,)))
    object_list.append(fiber_mon)

    if with_field_map:
        # full x-z plane phasor at lam_c only (single wavelength keeps the
        # memory trivial) — the coupling-region field figure
        field_mon = fdtdx.PhasorDetector(
            name="field_mon",
            partial_grid_shape=(None, 1, None),
            wave_characters=(fdtdx.WaveCharacter(wavelength=cfg.lam_c * UM),),
            components=("Ey",),
            exact_interpolation=True,
        )
        constraints.append(field_mon.same_size(volume, axes=(0, 2)))
        constraints.append(field_mon.place_at_center(volume, axes=(0, 1, 2)))
        object_list.append(field_mon)

    return sim_config, object_list, constraints


# --------------------------------------------------------------------------
# Measurement (Config B: fiber-side excitation)
# --------------------------------------------------------------------------




def _fdtd_forward(arrays, objects, sim_config, key):
    """Forward FDTD dispatch: the vendored fast loop (bitwise-gated, 1.79x)
    unless INVDX_FAST=0 or the scene is outside its supported subset, in
    which case fall back to vanilla fdtdx.run_fdtd with a notice."""
    import os

    if os.environ.get("INVDX_FAST", "1") != "0":
        try:
            from ..engines.fdtdx_perf import run_fdtd_fast

            return run_fdtd_fast(arrays=arrays, objects=objects,
                                 config=sim_config, key=key)
        except NotImplementedError as e:
            print(f"[fdtdx_perf] vanilla fallback: {e}")
    import fdtdx as _f

    return _f.run_fdtd(arrays=arrays, objects=objects, config=sim_config,
                       key=key)


def _run(cfg, teeth, with_chip, seed=0, azimuth_sign=1.0, excitation="fiber",
         with_field_map=False, shallow_teeth=None):
    sim_config, objs, cons = build_scene(cfg, teeth=teeth, with_chip=with_chip,
                                         azimuth_sign=azimuth_sign,
                                         excitation=excitation,
                                         with_field_map=with_field_map,
                                         shallow_teeth=shallow_teeth)
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, k2)
    _, arrays = _fdtd_forward(arrays, objects, sim_config, key)
    return arrays


def _phasor(arrays, name, comp_idx, y_axis, lam_idx=0):
    """Complex field line from a PhasorDetector: squeeze the singleton axes,
    then average the thin periodic y axis (fields are uniform along it)."""
    ph = np.squeeze(np.asarray(arrays.detector_states[name]["phasor"])[0, lam_idx, comp_idx])
    return ph.mean(axis=y_axis)


def beam_power_and_tilt(cfg, seed=0, azimuth_sign=1.0):
    """Empty-cell run: incident beam power AND the measured in-plane phase
    slope (rad/um) of the injected beam on the fiber line.

    The slope doubles as tilt calibration: the pinned fdtdx release maps
    azimuth to physical tilt with a sign that flips with grid resolution
    (verified 25nm vs 12.5nm at identical injected time offsets), so the
    tilt direction must be measured per configuration, never assumed.
    """
    arrays = _run(cfg, teeth=None, with_chip=False, seed=seed,
                  azimuth_sign=azimuth_sign)
    # fiber_mon is a z-plane: squeezed shape (nx, ny) -> y is axis 1
    Ey = _phasor(arrays, "fiber_mon", 0, y_axis=1)
    Hx = _phasor(arrays, "fiber_mon", 1, y_axis=1)
    p = phasor_line_power(Ey, Hx, cfg.spacing_um)

    n = Ey.shape[0]
    xs = (np.arange(n) - n / 2) * cfg.spacing_um
    m = np.abs(xs) < 3.0
    slope = float(np.polyfit(xs[m], np.unwrap(np.angle(Ey[m])), 1)[0])
    return p, slope


def calibrated_beam(cfg, seed=0):
    """Return (p_in, azimuth_sign, slope); azimuth stays -theta ALWAYS.

    Post-mortem of a wrong turn, kept as a warning: the SIGN of the measured
    fiber-line phase slope is a phasor time-reference convention that varies
    with run parameters (step count/wavelength list) while the injected
    time offsets — the physical beam — stay identical. Flipping the azimuth
    to "fix" the slope sign physically breaks the beam (res-80 calibration
    incident, 2026-07-13). Only the slope MAGNITUDE is a valid check:
    |slope| must equal k0*sin(theta).
    """
    p_in, slope = beam_power_and_tilt(cfg, seed=seed, azimuth_sign=1.0)
    if abs(cfg.theta_deg) > 0.1:
        k0 = 2 * np.pi / cfg.lam_c
        expected = k0 * np.sin(np.deg2rad(abs(cfg.theta_deg)))
        if abs(abs(slope) - expected) / expected > 0.05:
            raise RuntimeError(
                f"injected tilt magnitude wrong: |slope|={abs(slope):.3f} "
                f"rad/um vs k0*sin(theta)={expected:.3f} — beam injection "
                f"is broken, do not trust CE from this configuration")
    return p_in, 1.0, slope


def characterize(cfg, teeth, p_in=None, azimuth_sign=None, seed=0):
    """CE into the -x traveling slab TE0 mode (Config B). Returns dict."""
    slope = None
    if p_in is None or azimuth_sign is None:
        p_in, azimuth_sign, slope = calibrated_beam(cfg, seed=seed)
    arrays = _run(cfg, teeth=teeth, with_chip=True, seed=seed,
                  azimuth_sign=azimuth_sign)

    # wg_mon is an x-plane: squeezed shape (ny, nz) -> y is axis 0
    Ey = _phasor(arrays, "wg_mon", 0, y_axis=0)
    Hz = _phasor(arrays, "wg_mon", 1, y_axis=0)
    n = Ey.shape[0]
    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    zs = z_mon_lo + (np.arange(n) + 0.5) * cfg.spacing_um
    Em, Hm_fwd, neff = slab_te0_mode(zs, 0.0, cfg)
    # -x traveling mode: Hz flips sign
    p_mode = overlap_power_directional(Ey, Hz, Em, -Hm_fwd, cfg.spacing_um)
    ce = p_mode / p_in
    return {"CE": ce, "CE_dB": 10 * np.log10(ce + 1e-15),
            "P_in": p_in, "P_mode": p_mode, "n_eff": float(neff),
            "azimuth_sign": azimuth_sign, "tilt_slope_rad_per_um": slope}


# --------------------------------------------------------------------------
# 3D validation (extrude the 1D profile; pvgc validate_3d equivalent on GPU)
# --------------------------------------------------------------------------


def build_scene_3d(cfg, teeth, wg_width_um=10.0, azimuth_sign=1.0,
                   with_chip=True, with_field_map=False):
    """3D scene: the quasi-2D layout extruded to a straight grating of width
    W (y), radial Gaussian beam from above, PML on all six sides.

    Same pvgc coordinates; y is now real (W + 2*1.5um margins + 2*dpml).
    """
    import jax.numpy as jnp
    from ..engines.fdtdx_fixes import GaussianBeamSource

    spacing = cfg.spacing_um * UM
    cell_y = wg_width_um + 3.0 + 2 * cfg.dpml
    sim_config = fdtdx.SimulationConfig(
        time=cfg.sim_time_s, resolution=spacing,
        dtype=jnp.float32, courant_factor=0.99)

    object_list, constraints = [], []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(cfg.cell_x * UM, cell_y * UM, cfg.cell_z * UM))
    object_list.append(volume)

    npml = _pml_cells(cfg)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=npml)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(bound_dict.values())

    si = fdtdx.Material(permittivity=cfg.n_si ** 2)
    sio2 = fdtdx.Material(permittivity=cfg.n_sio2 ** 2)

    def block(name, material, x_min, x_size, z_min, z_size, y_width=None):
        """Block in pvgc coords; full y unless y_width (centered) given."""
        y_shape = None if y_width is None else y_width * UM
        obj = fdtdx.UniformMaterialObject(
            name=name, material=material,
            partial_real_shape=(x_size * UM, y_shape, z_size * UM))
        axes, own, other = (0, 2), (-1, -1), (-1, -1)
        margins = ((cfg.X0 + x_min) * UM, (cfg.Z0 + z_min) * UM)
        constraints.append(obj.place_relative_to(
            volume, axes=axes, own_positions=own, other_positions=other,
            margins=margins))
        if y_width is None:
            constraints.append(obj.same_size(volume, axes=(1,)))
        else:
            constraints.append(obj.place_at_center(volume, axes=(1,)))
        object_list.append(obj)

    if with_chip:
        block("substrate", si, -cfg.X0, cfg.cell_x,
              -(cfg.t_box + cfg.t_sub + cfg.dpml), cfg.t_sub + cfg.dpml)
        block("box", sio2, -cfg.X0, cfg.cell_x, -cfg.t_box, cfg.t_box)
        block("wg_slab", si, -cfg.X0, cfg.X0 - cfg.L_design / 2,
              0.0, cfg.t_si, y_width=wg_width_um)
        for k, (x_min, w) in enumerate(teeth or []):
            block(f"tooth{k}", si, x_min, w, 0.0, cfg.t_si,
                  y_width=wg_width_um)

    src_width = cfg.L_design + 2 * cfg.pad_x - 1
    src_width_y = cell_y - 2 * cfg.dpml - 1   # pvgc validate_3d aperture
    source = GaussianBeamSource(
        name="beam",
        partial_real_shape=(src_width * UM, src_width_y * UM, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=cfg.lam_c * UM),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_wave=fdtdx.WaveCharacter(wavelength=cfg.lam_c * UM),
            spectral_width=fdtdx.WaveCharacter(
                frequency=0.2 * 2.998e14 / cfg.lam_c)),
        direction="-",
        azimuth_angle=azimuth_sign * (-cfg.theta_deg),
        fixed_E_polarization_vector=(0.0, 1.0, 0.0),
        waist_radius=cfg.w0 * UM,
        profile_axis="radial",
    )
    constraints.append(source.place_relative_to(
        volume, axes=(0, 2), own_positions=(-1, -1), other_positions=(-1, -1),
        margins=(((cfg.X0 + cfg.fiber_x0 - src_width / 2)) * UM,
                 (cfg.Z0 + cfg.src_beam_y) * UM)))
    constraints.append(source.place_at_center(volume, axes=(1,)))
    object_list.append(source)

    lams_um = getattr(cfg, "_lams_um", None) or (cfg.lam_c,)
    wave_chars = tuple(fdtdx.WaveCharacter(wavelength=l * UM) for l in lams_um)

    wg_mon = fdtdx.PhasorDetector(
        name="wg_mon",
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, (wg_width_um + 2) * UM,
                            cfg.wg_mon_height * UM),
        wave_characters=wave_chars,
        components=("Ey", "Hz"),
        exact_interpolation=True,
    )
    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    constraints.append(wg_mon.place_relative_to(
        volume, axes=(0, 2), own_positions=(-1, -1), other_positions=(-1, -1),
        margins=((cfg.X0 + cfg.x_mon_wg) * UM, (cfg.Z0 + z_mon_lo) * UM)))
    constraints.append(wg_mon.place_at_center(volume, axes=(1,)))
    object_list.append(wg_mon)

    fib_width = cfg.cell_x - 2 * cfg.dpml - 0.5
    fiber_mon = fdtdx.PhasorDetector(
        name="fiber_mon",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(fib_width * UM, (cell_y - 2 * cfg.dpml - 0.5) * UM,
                            None),
        wave_characters=wave_chars,
        components=("Ey", "Hx"),
        exact_interpolation=True,
    )
    constraints.append(fiber_mon.place_relative_to(
        volume, axes=(0, 2), own_positions=(-1, -1), other_positions=(-1, -1),
        margins=(((cfg.cell_x - fib_width) / 2) * UM,
                 (cfg.Z0 + cfg.fiber_line_y - 0.4) * UM)))
    constraints.append(fiber_mon.place_at_center(volume, axes=(1,)))
    object_list.append(fiber_mon)

    if with_field_map:
        # two single-lambda slices through the 3D volume:
        #   side view (x-z at y = 0)          — compare with the quasi-2D map
        #   top view  (x-y at mid-slab z)     — lateral spreading, the view
        #                                       only 3D can provide
        wc = (fdtdx.WaveCharacter(wavelength=cfg.lam_c * UM),)
        f_xz = fdtdx.PhasorDetector(
            name="field_xz", partial_grid_shape=(None, 1, None),
            wave_characters=wc, components=("Ey",), exact_interpolation=True)
        constraints.append(f_xz.same_size(volume, axes=(0, 2)))
        constraints.append(f_xz.place_at_center(volume, axes=(0, 1, 2)))
        object_list.append(f_xz)

        f_xy = fdtdx.PhasorDetector(
            name="field_xy", partial_grid_shape=(None, None, 1),
            wave_characters=wc, components=("Ey",), exact_interpolation=True)
        constraints.append(f_xy.same_size(volume, axes=(0, 1)))
        constraints.append(f_xy.place_relative_to(
            volume, axes=(2,), own_positions=(-1,), other_positions=(-1,),
            margins=((cfg.Z0 + cfg.t_si / 2) * UM,)))
        constraints.append(f_xy.place_at_center(volume, axes=(0, 1)))
        object_list.append(f_xy)

    return sim_config, object_list, constraints


def eps_grid_xy(cfg, teeth, wg_width_um, nx, ny, cell_y_um):
    """Top-view permittivity slice at mid-slab (structure overlay)."""
    xs = (np.arange(nx) + 0.5) * cfg.spacing_um - cfg.X0
    ys = (np.arange(ny) + 0.5) * cfg.spacing_um - cell_y_um / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    eps = np.ones((nx, ny))
    in_w = np.abs(Y) < wg_width_um / 2
    eps[in_w & (X < -cfg.L_design / 2)] = cfg.n_si ** 2
    for x0, w in (teeth or []):
        eps[in_w & (X >= x0) & (X < x0 + w)] = cfg.n_si ** 2
    return eps


def field_map_3d(cfg, teeth, wg_width_um=10.0, seed=0):
    """Steady-state Ey(lam_c) slices from ONE 3D run: side view (x-z, y=0)
    and top view (x-y, mid-slab). The top view shows lateral spreading —
    information the quasi-2D chain cannot produce at all."""
    sim_config, objs, cons = build_scene_3d(
        cfg, teeth=teeth, wg_width_um=wg_width_um, with_chip=True,
        with_field_map=True)
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, k2)
    _, arrays = _fdtd_forward(arrays, objects, sim_config, key)

    xz = np.squeeze(np.asarray(
        arrays.detector_states["field_xz"]["phasor"])[0, 0, 0])
    xy = np.squeeze(np.asarray(
        arrays.detector_states["field_xy"]["phasor"])[0, 0, 0])
    del arrays

    cell_y = wg_width_um + 3.0 + 2 * cfg.dpml
    out = {}
    out["xz"] = {"field": xz,
                 "eps": eps_grid_xz(cfg, teeth, *xz.shape),
                 "extent": np.array((-cfg.X0, cfg.X0, -cfg.Z0,
                                     cfg.cell_z - cfg.Z0)),
                 "title": f"3D 側視 Re Ey  (y=0, λ = {cfg.lam_c} µm)"}
    out["xy"] = {"field": xy,
                 "eps": eps_grid_xy(cfg, teeth, wg_width_um, *xy.shape,
                                    cell_y_um=cell_y),
                 "extent": np.array((-cfg.X0, cfg.X0, -cell_y / 2,
                                     cell_y / 2)),
                 "title": f"3D 俯視 Re Ey  (矽層中央, λ = {cfg.lam_c} µm)"}
    return out


def wg_mode_3d(ys_um, zs_um, cfg, wg_width_um=10.0):
    """Separable approximation of the fundamental mode of the W-wide,
    220nm-thick waveguide: TE0(z) * cos(pi*y/W_eff) lateral envelope.

    W_eff = W + 0.4um accounts for lateral field penetration. Good to a few
    percent in overlap for W >> lambda/n — validation-grade, and stated as
    an approximation wherever the resulting CE is reported.
    """
    Ez_prof, Hz_prof, neff = slab_te0_mode(zs_um, 0.0, cfg)
    W_eff = wg_width_um + 0.4
    lat = np.cos(np.pi * np.asarray(ys_um) / W_eff)
    lat = np.where(np.abs(ys_um) <= W_eff / 2, lat, 0.0)
    Em = lat[:, None] * Ez_prof[None, :]
    Hm = lat[:, None] * Hz_prof[None, :]
    return Em, Hm, neff


def run_3d_planes(cfg, teeth, wg_width_um, with_chip, lams_um, seed=0):
    """One 3D run -> host-side phasor planes {'fiber': ..., 'wg': ...}.

    The unit of dual-GPU task parallelism: each of the two runs of a 3D
    measurement (empty-cell normalization / grating) is independent, so they
    can execute on different devices in different processes and be combined
    afterwards (combine_3d).
    """
    cfg._lams_um = tuple(lams_um) if lams_um else (cfg.lam_c,)
    try:
        arrays = _run_3d(cfg, teeth, wg_width_um, seed, with_chip=with_chip)
        planes = {
            "fiber": np.asarray(
                arrays.detector_states["fiber_mon"]["phasor"]).copy(),
            "wg": np.asarray(
                arrays.detector_states["wg_mon"]["phasor"]).copy(),
        }
        del arrays
        return planes
    finally:
        cfg._lams_um = None


def combine_3d(fiber_planes_empty, wg_planes_grating, cfg, lams_um,
               wg_width_um=10.0):
    """CE(lambda) from the empty-run fiber planes + grating-run wg planes."""
    dA = cfg.spacing_um ** 2
    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    spectrum = []
    for k, lam in enumerate(lams_um):
        Ey = np.squeeze(fiber_planes_empty[0, k, 0])
        Hx = np.squeeze(fiber_planes_empty[0, k, 1])
        p_in = float(abs(0.5 * np.real(np.sum(Ey * np.conj(Hx))) * dA))

        Ey_m = np.squeeze(wg_planes_grating[0, k, 0])
        Hz_m = np.squeeze(wg_planes_grating[0, k, 1])
        ny, nz = Ey_m.shape
        zs = z_mon_lo + (np.arange(nz) + 0.5) * cfg.spacing_um
        ys = (np.arange(ny) - ny / 2 + 0.5) * cfg.spacing_um
        lam_cfg = type(cfg)()
        lam_cfg.lam_c, lam_cfg.t_si = lam, cfg.t_si
        lam_cfg.n_si, lam_cfg.n_sio2 = cfg.n_si, cfg.n_sio2
        Em, Hm_fwd, neff = wg_mode_3d(ys, zs, lam_cfg, wg_width_um)
        p_mode = overlap_power_directional(Ey_m, Hz_m, Em, -Hm_fwd, dA)
        ce = p_mode / p_in
        spectrum.append({"lam_um": float(lam), "CE": float(ce),
                         "CE_dB": float(10 * np.log10(abs(ce) + 1e-15)),
                         "P_in": p_in, "n_eff": float(neff)})
    return {"spectrum": spectrum,
            "peak": max(spectrum, key=lambda r: r["CE_dB"]),
            "note": "wg mode is a separable TE0(z)*cos(y) approximation"}


def characterize_3d(cfg, teeth, wg_width_um=10.0, lams_um=None, seed=0):
    """3D CE (Config B, vertical/tilted radial beam) — pvgc validate_3d
    equivalent. Two runs: empty-cell beam power + grating run. When lams_um
    is given, CE(lambda) comes from the same two runs (multi-wavelength
    phasors) — always prefer the spectrum, per conventions lesson 6.
    """
    dA = cfg.spacing_um ** 2
    lams = tuple(lams_um) if lams_um else (cfg.lam_c,)
    cfg._lams_um = lams
    try:
        # Sequential runs must NOT hold both ArrayContainers: each carries
        # the full-volume E/H/PML arrays and two of them OOM a 48GB card at
        # 3D sizes. Copy the (tiny) phasor planes to host, drop the
        # container, then run the next simulation.
        arrays_e = _run_3d(cfg, None, wg_width_um, seed, with_chip=False)
        fiber_planes = np.asarray(
            arrays_e.detector_states["fiber_mon"]["phasor"]).copy()
        del arrays_e

        arrays_g = _run_3d(cfg, teeth, wg_width_um, seed, with_chip=True)
        wg_planes = np.asarray(
            arrays_g.detector_states["wg_mon"]["phasor"]).copy()
        del arrays_g
    finally:
        cfg._lams_um = None

    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    spectrum = []
    for k, lam in enumerate(lams):
        Ey = np.squeeze(fiber_planes[0, k, 0])
        Hx = np.squeeze(fiber_planes[0, k, 1])
        p_in = float(abs(0.5 * np.real(np.sum(Ey * np.conj(Hx))) * dA))

        Ey_m = np.squeeze(wg_planes[0, k, 0])
        Hz_m = np.squeeze(wg_planes[0, k, 1])
        ny, nz = Ey_m.shape
        zs = z_mon_lo + (np.arange(nz) + 0.5) * cfg.spacing_um
        ys = (np.arange(ny) - ny / 2 + 0.5) * cfg.spacing_um
        lam_cfg = type(cfg)()
        lam_cfg.lam_c, lam_cfg.t_si = lam, cfg.t_si
        lam_cfg.n_si, lam_cfg.n_sio2 = cfg.n_si, cfg.n_sio2
        Em, Hm_fwd, neff = wg_mode_3d(ys, zs, lam_cfg, wg_width_um)
        p_mode = overlap_power_directional(Ey_m, Hz_m, Em, -Hm_fwd, dA)
        ce = p_mode / p_in
        spectrum.append({"lam_um": float(lam), "CE": float(ce),
                         "CE_dB": float(10 * np.log10(abs(ce) + 1e-15)),
                         "P_in": p_in, "n_eff": float(neff)})
    out = dict(spectrum[0]) if len(spectrum) == 1 else {
        "spectrum": spectrum,
        "peak": max(spectrum, key=lambda r: r["CE_dB"])}
    out["note"] = "wg mode is a separable TE0(z)*cos(y) approximation"
    return out


def _run_3d(cfg, teeth, wg_width_um, seed, with_chip):
    sim_config, objs, cons = build_scene_3d(
        cfg, teeth=teeth, wg_width_um=wg_width_um, with_chip=with_chip)
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, k2)
    _, arrays = _fdtd_forward(arrays, objects, sim_config, key)
    return arrays


def _phasor3(arrays, name, comp_idx, lam_idx=0):
    """2D complex field plane from a PhasorDetector in the 3D scene."""
    return np.squeeze(
        np.asarray(arrays.detector_states[name]["phasor"])[0, lam_idx, comp_idx])


def eps_grid_xz(cfg, teeth, nx, nz, shallow_teeth=None):
    """Rasterized permittivity of the x-z cross-section (structure overlay
    for field maps; coordinates match the field_mon plane)."""
    xs = (np.arange(nx) + 0.5) * cfg.spacing_um - cfg.X0   # pvgc x
    zs = (np.arange(nz) + 0.5) * cfg.spacing_um - cfg.Z0   # pvgc y
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    eps = np.ones((nx, nz))
    eps[Z < -cfg.t_box] = cfg.n_si ** 2                    # substrate (+PML)
    eps[(Z >= -cfg.t_box) & (Z < 0)] = cfg.n_sio2 ** 2     # BOX
    in_si = (Z >= 0) & (Z < cfg.t_si)
    eps[in_si & (X < -cfg.L_design / 2)] = cfg.n_si ** 2   # wg slab
    for x0, w in (teeth or []):
        eps[in_si & (X >= x0) & (X < x0 + w)] = cfg.n_si ** 2
    in_shallow = (Z >= 0) & (Z < cfg.t_si - cfg.t_shallow)
    for x0, w in (shallow_teeth or []):
        eps[in_shallow & (X >= x0) & (X < x0 + w)] = cfg.n_si ** 2
    return eps


def field_map(cfg, teeth, seed=0, shallow_teeth=None):
    """Steady-state Ey(lam_c) over the full x-z plane, fiber-side excitation
    (the coupling-region figure: beam coming down, light leaving along the
    waveguide). One extra simulation; returns field + structure overlay."""
    arrays = _run(cfg, teeth=teeth, with_chip=True, seed=seed,
                  with_field_map=True, shallow_teeth=shallow_teeth)
    field = np.squeeze(np.asarray(
        arrays.detector_states["field_mon"]["phasor"])[0, 0, 0])
    nx, nz = field.shape
    eps = eps_grid_xz(cfg, teeth, nx, nz, shallow_teeth=shallow_teeth)
    extent = (-cfg.X0, cfg.X0, -cfg.Z0, cfg.cell_z - cfg.Z0)
    return {"field": field, "eps": eps, "extent": np.array(extent),
            "title": f"耦合區穩態場 Re Ey  (λ = {cfg.lam_c} µm)"}


def gaussian_mode_tilted(xs_um, x0, w0, lam_um, theta_deg, kx_sign=-1.0):
    """Tilted upward Gaussian target on a horizontal line (complex fields).

    Eg = exp(-((x-x0)/w0)^2) * exp(i * kx_sign * k0 * sin(theta) * (x-x0)),
    Hg = -cos(theta) * Eg — the MINUS marks upward (+z) propagation: for
    E along y, an upward wave has Hx = -Ey while a downward one has +Ey
    (verified on the injected Config B beam). Getting this sign wrong makes
    the directional overlap reject the real signal entirely (~ -60 dB).
    kx_sign=-1 is the reciprocal partner of the incoming Config B beam
    (time reversal flips the lateral k of the incident tilt).
    """
    xs = np.asarray(xs_um)
    k0 = 2 * np.pi / lam_um
    th = np.deg2rad(theta_deg)
    env = np.exp(-(((xs - x0) / w0) ** 2))
    Eg = env * np.exp(1j * kx_sign * k0 * np.sin(th) * (xs - x0))
    Hg = -np.cos(th) * Eg
    return Eg, Hg


def wg_side_characterize(cfg, teeth, seed=0, shallow_teeth=None):
    """Config A: waveguide-side excitation, ONE run -> (CE_fwd, S11, P_in).

    P_in  = forward TE0 overlap at wg_mon (filters injection impurity,
            pvgc's a_fwd pattern)
    S11   = backward TE0 overlap / P_in
    CE_fwd= Gaussian overlap on the fiber line / P_in; because the target is
            complex when tilted and the phasor time-reference convention
            varies between run configurations (see calibrated_beam), CE is
            evaluated for both kx signs and both reported.
    """
    arrays = _run(cfg, teeth=teeth, with_chip=True, seed=seed,
                  excitation="wg", shallow_teeth=shallow_teeth)

    # wg monitor: forward = P_in, backward = reflection
    Ey = _phasor(arrays, "wg_mon", 0, y_axis=0)
    Hz = _phasor(arrays, "wg_mon", 1, y_axis=0)
    n = Ey.shape[0]
    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    zs = z_mon_lo + (np.arange(n) + 0.5) * cfg.spacing_um
    Em, Hm_fwd, neff = slab_te0_mode(zs, 0.0, cfg)
    p_in = overlap_power_directional(Ey, Hz, Em, Hm_fwd, cfg.spacing_um)
    p_back = overlap_power_directional(Ey, Hz, Em, -Hm_fwd, cfg.spacing_um)
    s11 = p_back / p_in

    # fiber line: upward power into the (tilted) Gaussian target
    Ey_f = _phasor(arrays, "fiber_mon", 0, y_axis=1)
    Hx_f = _phasor(arrays, "fiber_mon", 1, y_axis=1)
    nf = Ey_f.shape[0]
    # fiber_mon is centered on the cell center = pvgc x=0, so these ARE pvgc
    # x-coordinates; gaussian_mode_tilted then centers the target at fiber_x0
    # (adding fiber_x0 here would cancel that shift and pin the target to
    # x=0 no matter where the fiber is)
    xs = (np.arange(nf) - nf / 2) * cfg.spacing_um
    ces = {}
    for kx_sign in (-1.0, +1.0):
        Eg, Hg = gaussian_mode_tilted(xs, cfg.fiber_x0, cfg.w0, cfg.lam_c,
                                      cfg.theta_deg, kx_sign=kx_sign)
        p_g = overlap_power_directional(Ey_f, Hx_f, Eg, Hg, cfg.spacing_um)
        ces[kx_sign] = p_g / p_in
    ce = max(ces.values())
    return {"CE_fwd": float(ce),
            "CE_fwd_dB": float(10 * np.log10(ce + 1e-15)),
            "CE_fwd_both_signs_dB": {str(k): float(10 * np.log10(v + 1e-15))
                                     for k, v in ces.items()},
            "S11": float(s11), "S11_dB": float(10 * np.log10(s11 + 1e-15)),
            "P_in": float(p_in), "n_eff": float(neff)}


def bandwidth_3db(spectrum):
    """(bw_um, lam_lo, lam_hi, note) — 3 dB bandwidth of a CE(lambda) list.

    Crossings are linearly interpolated between samples. When the spectrum
    does not drop 3 dB below the peak inside the sampled range, the affected
    edge is clamped to the range end and the note says so — the bandwidth is
    then a LOWER BOUND, never silently exact (paper honesty rule).
    """
    lam = np.array([r["lam_um"] for r in spectrum])
    ce = np.array([r["CE_dB"] for r in spectrum])
    k = int(np.argmax(ce))
    thr = ce[k] - 3.0

    def cross(idx_range, lo_side):
        prev_l, prev_c = lam[k], ce[k]
        for i in idx_range:
            if ce[i] < thr:
                f = (prev_c - thr) / (prev_c - ce[i])
                return prev_l + f * (lam[i] - prev_l), False
            prev_l, prev_c = lam[i], ce[i]
        return (lam[0], True) if lo_side else (lam[-1], True)

    lam_lo, clip_lo = cross(range(k - 1, -1, -1), True)
    lam_hi, clip_hi = cross(range(k + 1, len(lam)), False)
    note = None
    if clip_lo or clip_hi:
        side = "both edges" if clip_lo and clip_hi else (
            "lower edge" if clip_lo else "upper edge")
        note = (f"{side} clipped by the sampled range — bandwidth is a "
                f"lower bound; widen the lambda span to resolve it")
    return float(lam_hi - lam_lo), float(lam_lo), float(lam_hi), note


def characterize_spectrum(cfg, teeth, lams_um, azimuth_sign=None, seed=0,
                          shallow_teeth=None):
    """CE(lambda) over a dense wavelength list from ONE grating run + ONE
    empty-cell run (multi-wavelength phasors). The v2 lesson operationalized:
    dense spectra must be cheap or they get skipped.

    Slab-mode dispersion is handled per wavelength (n_eff(lam) re-solved).
    """
    # calibrate BEFORE arming multi-wavelength phasors: the slope check
    # compares against k0*sin(theta) at lam_c, so the calibration run must
    # measure at lam_c (with _lams_um set it would measure lams_um[0] and
    # spuriously fail for lists starting far from lam_c)
    if azimuth_sign is None:
        _, azimuth_sign, _ = calibrated_beam(cfg, seed=seed)
    cfg._lams_um = tuple(lams_um)
    try:
        arrays_e = _run(cfg, teeth=None, with_chip=False, seed=seed,
                        azimuth_sign=azimuth_sign)
        arrays_g = _run(cfg, teeth=teeth, with_chip=True, seed=seed,
                        azimuth_sign=azimuth_sign,
                        shallow_teeth=shallow_teeth)
    finally:
        cfg._lams_um = None

    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    out = []
    for k, lam in enumerate(lams_um):
        Ey_e = _phasor(arrays_e, "fiber_mon", 0, y_axis=1, lam_idx=k)
        Hx_e = _phasor(arrays_e, "fiber_mon", 1, y_axis=1, lam_idx=k)
        p_in = phasor_line_power(Ey_e, Hx_e, cfg.spacing_um)

        Ey = _phasor(arrays_g, "wg_mon", 0, y_axis=0, lam_idx=k)
        Hz = _phasor(arrays_g, "wg_mon", 1, y_axis=0, lam_idx=k)
        n = Ey.shape[0]
        zs = z_mon_lo + (np.arange(n) + 0.5) * cfg.spacing_um
        lam_cfg = type(cfg)(**{})  # only lam-dependent mode params matter
        lam_cfg.lam_c, lam_cfg.t_si = lam, cfg.t_si
        lam_cfg.n_si, lam_cfg.n_sio2 = cfg.n_si, cfg.n_sio2
        Em, Hm_fwd, neff = slab_te0_mode(zs, 0.0, lam_cfg)
        p_mode = overlap_power_directional(Ey, Hz, Em, -Hm_fwd, cfg.spacing_um)
        ce = p_mode / p_in
        out.append({"lam_um": float(lam), "CE": float(ce),
                    "CE_dB": float(10 * np.log10(abs(ce) + 1e-15)),
                    "P_in": p_in, "n_eff": float(neff)})
    return {"spectrum": out, "azimuth_sign": azimuth_sign}
