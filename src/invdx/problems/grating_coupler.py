"""grating_coupler problem — a fiber-to-chip grating coupler on a silicon-on-insulator
stack, built on the fdtdx engine. The scene this module assembles is the
fiber-side excitation measurement:

    fiber-side Gaussian beam (angle theta) -> CE into the -x traveling slab
    TE0 mode, normalized by the incident beam power from an empty-cell run.

Coordinate mapping (grating_coupler 2D -> fdtdx quasi-2D):
    grating_coupler x (propagation) -> fdtdx x, offset +X0 so the cell starts at 0
    grating_coupler y (vertical)    -> fdtdx z, offset +Z0 (grating_coupler y=0 = Si layer bottom)
    out-of-plane         -> fdtdx y, single cell, periodic boundaries
    grating_coupler Ez polarization -> fdtdx E_y

Field-convention notes (fdtdx stores eta0-normalized H, so H carries E units):
    plane wave in air, -z propagation:  Hx =  Ey
    slab TE0 mode, +x propagation:      Hz =  n_eff * Ey  (-x: flip sign)
Both match the natural-unit relations invdx.modes assumes, so the same
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
from ..engines.fdtdx_checkpoint_buffers import run_fdtd_buffers
from .contract import GradcheckCase, ProblemSpec, ReciprocityCase

UM = 1e-6


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
# Scene construction (fdtdx, quasi-2D)
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


def _pml_cells(cfg):
    return int(round(cfg.dpml / cfg.spacing_um))


def _require_float32_dtype(cfg):
    """build_scene/build_scene_3d hardcode fdtdx.SimulationConfig(dtype=
    jnp.float32); cfg.dtype is otherwise only read by
    engines.fdtdx_engine.make_sim_config. Rather than silently ignoring a
    cfg.dtype="float64" request, fail loudly: float64 through this fdtdx/jax
    pipeline has never been exercised here and may error outright or merely
    double memory/runtime with no verified accuracy gain. To add float64
    support, thread cfg.dtype into both SimulationConfig(dtype=...) call
    sites below and validate forward+adjoint correctness before trusting it.
    """
    if cfg.dtype != "float32":
        raise NotImplementedError(
            f"cfg.dtype={cfg.dtype!r} is not honored by build_scene/"
            f"build_scene_3d: this path hardcodes float32 (see "
            f"grating_coupler._require_float32_dtype). Only "
            f"engines.fdtdx_engine.make_sim_config reads cfg.dtype today.")


def _box_bounds(cfg):
    """energy_budget()'s closure-box extents, in grating_coupler (x, y=vertical) um.

    Judgment #3 (abstention): the "four outward faces sum to zero" identity
    only holds for a box whose INTERIOR has neither loss nor an active
    source. This project's materials are always constant-real-permittivity
    Si/SiO2/air (no lossy/dispersive material ever enters cfg), so the only
    way the premise can break is geometric: a face landing inside the PML,
    or the box enclosing the fiber-side source. Both are checked below and
    RAISE rather than silently returning a number that merely looks
    plausible (this is exactly the "wrong tool, no error" trap judgment #1
    warns about, applied to geometry instead of detector choice).

    inset=0.25 um reuses `fiber_mon`'s own inset from the PML boundary (see
    build_scene: fib_width = cell_x - 2*dpml - 0.5), so the box's top face
    is deliberately IDENTICAL to fiber_mon's plane — face_out_zhi in
    energy_budget() reads fiber_mon directly instead of a duplicate
    detector that could silently disagree with it.

    Note on what is/isn't checked here: x_lo/x_hi/z_lo are each defined as
    "PML boundary minus `inset`", so they clear the x/bottom-z PML by
    construction for ANY cfg with a non-degenerate interior — checking them
    against the PML boundary again would be a tautology (inset > 0). The
    one PML-adjacent bound that is NOT structurally self-clearing is z_hi:
    it comes from fiber_line_y, a parameter with no algebraic relationship
    to the top PML boundary (air_above), so it genuinely can land inside
    the PML for an unusual cfg — that case is checked explicitly below.
    The degenerate-box check at the end is what catches interiors too
    small to hold `inset` at all (equivalent to "PML has eaten the box").
    """
    inset = 0.25
    x_lo = -(cfg.X0 - cfg.dpml - inset)
    x_hi = +(cfg.X0 - cfg.dpml - inset)
    z_lo = -(cfg.t_box + cfg.t_sub) + inset
    z_hi = cfg.fiber_line_y - 0.4   # == fiber_mon's z-plane, see above

    interior_z_hi = cfg.cell_z - cfg.Z0 - cfg.dpml   # == t_si + air_above
    if z_hi >= interior_z_hi - 0.05:
        raise ValueError(
            f"energy_budget box top face (z_hi={z_hi:.4g} um) does not "
            f"clear the top PML boundary at {interior_z_hi:.4g} um — "
            f"refusing to report a closure check whose no-loss-inside "
            f"premise (judgment #3) would be violated by absorbing PML "
            f"cells. Raise air_above or lower fiber_line_y.")
    if z_hi >= cfg.src_beam_y - 0.05:
        raise ValueError(
            f"energy_budget box top face (z_hi={z_hi:.4g} um) is not "
            f"safely below the fiber-side source plane (src_beam_y="
            f"{cfg.src_beam_y} um): the box would enclose an active "
            f"source, and Poynting's theorem then no longer implies zero "
            f"net outward flux for a lossless interior (a source "
            f"contributes real injected power, not zero) — the closure "
            f"check's premise would be violated silently. Raise "
            f"src_beam_y or lower fiber_line_y so the box clears it.")
    if x_lo >= x_hi or z_lo >= z_hi:
        raise ValueError(
            f"energy_budget box is degenerate for this cfg "
            f"(x=[{x_lo:.4g},{x_hi:.4g}], z=[{z_lo:.4g},{z_hi:.4g}] um) — "
            f"check dpml/X0/t_box/t_sub/fiber_line_y.")
    return x_lo, x_hi, z_lo, z_hi


def build_scene(cfg, teeth=None, with_chip=True, azimuth_sign=1.0,
                excitation="fiber", with_field_map=False,
                shallow_teeth=None, with_energy_box=False):
    """Object list + constraints for one run.

    teeth        — list of (x_min_um, width_um) Si teeth in grating_coupler coordinates
                   (None = no grating; wg slab still present when with_chip)
    shallow_teeth— optional second-layer regions: Si of height
                   t_si - t_shallow (a 70 nm shallow etch leaves 150 nm).
                   Layers are independent patterns; where a full tooth and a
                   shallow region overlap, full silicon wins (blocks simply
                   stack)
    with_chip    — False gives the empty air cell that measures the incident
                   beam power (the CE denominator)
    excitation   — "fiber": fiber-side excitation, Gaussian beam from above
                   "wg": waveguide-side excitation, +x beam launched inside
                   the waveguide; injection impurity is normalized away by
                   the forward mode overlap at wg_mon
    azimuth_sign — diagnostic override of the fiber-beam tilt direction;
                   production code leaves it at 1.0 (azimuth = -theta)

    Detectors always present:
      "wg_mon"    x-plane at cfg.x_mon_wg, components (Ey, Hz)
      "fiber_mon" z-plane 0.4 um below the fiber line, components (Ey, Hx)
    with_energy_box adds three more PhasorDetector lines ("ebox_xlo",
      "ebox_xhi", "ebox_zlo") forming, together with the always-present
      "fiber_mon", the four faces of `energy_budget`'s closure box (see
      `_box_bounds` and `energy_budget`) — all PhasorDetector, never
      fdtdx.PoyntingFluxDetector (judgment #1: that detector is an
      instantaneous time-domain quantity and is meaningless mixed with a
      pulsed source + phasor CE).
    Returns (config, object_list, constraints).
    """
    import jax.numpy as jnp

    _require_float32_dtype(cfg)
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
        """Si/SiO2 block in grating_coupler coordinates, spanning the full thin y axis."""
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
        # substrate Si extends through the bottom PML
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
    # engines/fdtdx_fixes.py). Amplitude is exp(-((x-x0)/w0)^2).
    from ..engines.fdtdx_fixes import GaussianBeamSource

    if excitation == "wg":
        # waveguide-side excitation: +x beam inside the waveguide core
        # (z-Gaussian profile)
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
    # single run (the sparse-sampling failure mode below makes cheap dense
    # spectra a first-class need)
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

    if with_energy_box:
        # energy_budget()'s closure box: three NEW line detectors (the top
        # face reuses fiber_mon exactly — same plane, same width — rather
        # than adding a fourth detector that could silently disagree with
        # it). All PhasorDetector (judgment #1); components/orientation
        # match wg_mon (x-faces: Ey,Hz) / fiber_mon (z-face: Ey,Hx).
        x_lo, x_hi, z_lo, z_hi = _box_bounds(cfg)
        for name, x_at in (("ebox_xlo", x_lo), ("ebox_xhi", x_hi)):
            ebx = fdtdx.PhasorDetector(
                name=name,
                partial_grid_shape=(1, None, None),
                partial_real_shape=(None, None, (z_hi - z_lo) * UM),
                wave_characters=wave_chars,
                components=("Ey", "Hz"),
                exact_interpolation=True,
            )
            constraints.append(ebx.place_relative_to(
                volume, axes=(0, 2), own_positions=(-1, -1),
                other_positions=(-1, -1),
                margins=((cfg.X0 + x_at) * UM, (cfg.Z0 + z_lo) * UM)))
            constraints.append(ebx.same_size(volume, axes=(1,)))
            object_list.append(ebx)

        ebox_zlo = fdtdx.PhasorDetector(
            name="ebox_zlo",
            partial_grid_shape=(None, None, 1),
            partial_real_shape=((x_hi - x_lo) * UM, None, None),
            wave_characters=wave_chars,
            components=("Ey", "Hx"),
            exact_interpolation=True,
        )
        constraints.append(ebox_zlo.place_relative_to(
            volume, axes=(0, 2), own_positions=(-1, -1),
            other_positions=(-1, -1),
            margins=((cfg.X0 + x_lo) * UM, (cfg.Z0 + z_lo) * UM)))
        constraints.append(ebox_zlo.same_size(volume, axes=(1,)))
        object_list.append(ebox_zlo)

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
# Measurement (fiber-side excitation)
# --------------------------------------------------------------------------




def _fdtd_forward(arrays, objects, sim_config, key):
    """Forward FDTD dispatch: the vendored fast loop (bitwise-gated, ~2x)
    unless INVDX_FAST=0 or the scene is outside its supported subset, in
    which case fall back to vanilla fdtdx.run_fdtd with a notice.

    Memory reclaim is on by default (INVDX_FAST_RECLAIM=0 to disable): the
    fast loop frees the caller's full-volume psi/alpha/kappa/sigma/field
    buffers during the time loop, roughly tripling the grid size that fits
    on a GPU. Safe here because every grating_coupler runner immediately rebinds its
    `arrays` name to the returned container and never touches the input
    container again — keep it that way when adding runners, or the freed
    input arrays will raise on use."""
    import os

    if os.environ.get("INVDX_FAST", "1") != "0":
        try:
            from ..engines.fdtdx_perf import run_fdtd_fast

            reclaim = os.environ.get("INVDX_FAST_RECLAIM", "1") != "0"
            return run_fdtd_fast(arrays=arrays, objects=objects,
                                 config=sim_config, key=key,
                                 reclaim_memory=reclaim)
        except NotImplementedError as e:
            print(f"[fdtdx_perf] vanilla fallback: {e}")
    import fdtdx as _f

    return _f.run_fdtd(arrays=arrays, objects=objects, config=sim_config,
                       key=key)


def _run(cfg, teeth, with_chip, seed=0, azimuth_sign=1.0, excitation="fiber",
         with_field_map=False, shallow_teeth=None, with_energy_box=False):
    sim_config, objs, cons = build_scene(cfg, teeth=teeth, with_chip=with_chip,
                                         azimuth_sign=azimuth_sign,
                                         excitation=excitation,
                                         with_field_map=with_field_map,
                                         shallow_teeth=shallow_teeth,
                                         with_energy_box=with_energy_box)
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
    to "fix" the slope sign physically breaks the beam (seen for real during
    a resolution-calibration check). Only the slope MAGNITUDE is a valid
    check: |slope| must equal k0*sin(theta).
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
    """CE into the -x traveling slab TE0 mode, fiber-side excitation.

    Returns a dict.
    """
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
# Energy budget — "where did the power go", promoted to a first-class,
# conservation-checked measurement (fiber-side excitation only).
#
# Genesis: a manual power tally on this scene showed that CE factorizes
# into two independent stages — how much of P_in gets into the waveguide at
# all, and what fraction of THAT lands in the target TE0 mode — and that the
# first stage can be dominated by power radiating down into the substrate
# rather than by reflection. CE alone cannot tell those channels apart,
# which is the whole reason to keep the tally as a measurement instead of a
# one-off. It took five iterations to close (the first four summed to
# 144-151% because a tool's "flux" detector was returning an unsigned
# time-domain quantity, not a signed phasor one — an error invisible if you
# only ever look at CE). The five judgments below are what makes it not
# happen again; they are argued in more detail inline.
# --------------------------------------------------------------------------


def check_energy_closure(closure_residual_frac_of_input, hi=0.005,
                         face_terms=None, port_face_net_in=None,
                         ulp_factor=8.0):
    """Two-sided gate on |closure_residual_frac_of_input| (judgment #4).

    TOO BIG a residual (> hi, default 0.5%) is the obvious failure: the
    four box faces do not sum to zero, so a face is mis-signed,
    mis-placed, double-counted, or the box violates `_box_bounds`'s
    no-PML/no-source premise.

    TOO SMALL is also a failure, but the floor is now the arithmetic's own
    resolution rather than a fixed percentage. The concern is unchanged:
    a residual that cancels *too* cleanly is the signature of a bug that
    looks tidy — the same flux counted on two "different" faces, or a sign
    flip on one face exactly undoing a scale error on another.

    The earlier form of this gate used a fixed lower bound (0.1%), and that
    was wrong in a way no choice of number could fix. The residual is a
    CONVERGENT quantity: refine the grid and it approaches zero. Any fixed
    positive bound is therefore guaranteed to fire on a correct
    implementation — the only question is at which resolution. A
    grid-convergence study (four spacings from 0.080 down to 0.010 um) put
    production at 0.1505%, a factor of 1.5 from tripping its own gate, and
    extrapolating the measured order (2.26) says ~1 nm cells would cross
    any 1e-5 bound. Lowering the number would only move the false alarm to
    a finer grid, which in the short term is indistinguishable from having
    fixed it. The mistake was expressing a question about whether the
    implementation is honest along the axis of resolution; the more
    correct the usage, the more surely it fired.

    What does survive that argument: a residual computed in floating point
    approaches the arithmetic noise floor but does not pass through it, so
    a value at or below a few ulp of the largest term in the sum cannot
    come from a fine grid. It can only come from algebraic cancellation --
    which is exactly the enforced-conservation bug. Pass `face_terms` (the
    four signed face fluxes) and `port_face_net_in` to get that floor;
    without them the gate falls back to flagging only an exactly-zero
    residual, which is the same argument at its weakest useful setting.

    That floor is a KNOWN UNDERESTIMATE, and deliberately left as one. It
    scales with |face_out_*|, the flux left after the integrand has already
    cancelled along that face -- and on the top face Sx/Sz genuinely changes
    sign (radiation out, some flow back), so the roundoff actually
    accumulated there goes with sum(|integrand·dl|), which can be far larger
    than the net. A truer floor would need each face's sum-of-magnitudes
    passed in as well. Underestimating makes the gate harder to trip, i.e.
    duller but never false-alarming, which is the right direction for a
    check meant to be nearly silent. It is written down here because the
    failure mode is a later reader seeing it almost never fire, concluding
    it is too loose, and tightening a number that was already low.

    This gate is nearly silent by design, and that is its specification,
    not a defect -- it should fire only for the bug it names. The
    complementary check, that the accounting is sensitive at all, is a
    negative control that perturbs one face deliberately; see
    `tests/test_energy_budget.py`.

    Never raises (so it can run unattended inside a measurement pipeline,
    e.g. characterize()/scripts/07, without aborting a batch) — callers
    that want a hard stop should check the returned "ok" field.
    """
    r = abs(float(closure_residual_frac_of_input))
    if r > hi:
        return {"status": "fail_high", "ok": False, "residual_frac": r,
                "message": (
                    f"|closure_residual_frac_of_input| = {r:.4%} > "
                    f"{hi:.4%}: the four box faces do not sum to zero — "
                    f"the books do not balance. Check face signs/"
                    f"placement, or whether the box now clips PML or a "
                    f"source (see _box_bounds).")}

    floor = 0.0
    if face_terms is not None and port_face_net_in:
        scale = max(abs(float(t)) for t in face_terms)
        eps = float(np.finfo(np.float32).eps)
        floor = ulp_factor * eps * scale / abs(float(port_face_net_in))
    if r <= floor:
        return {"status": "fail_low", "ok": False, "residual_frac": r,
                "message": (
                    f"|closure_residual_frac_of_input| = {r:.6g} is at or "
                    f"below the arithmetic floor {floor:.6g} "
                    f"({ulp_factor:g} ulp of the largest face term). A "
                    f"residual cannot land there by having a fine enough "
                    f"grid — only by algebraic cancellation, which is what "
                    f"a double-counted face or a sign/scale bug that "
                    f"exactly undoes itself looks like. Re-check the box "
                    f"geometry before trusting this run.")}
    return {"status": "ok", "ok": True, "residual_frac": r,
            "message": (f"closure residual {r:.4%} above the arithmetic "
                        f"floor {floor:.3g} and within {hi:.4%}")}


def _energy_budget_from_fields(cfg, zs_port, Ey_port, Hz_port,
                               Ey_xlo, Hz_xlo, Ey_xhi, Hz_xhi,
                               Ey_zlo, Hx_zlo, Ey_zhi, Hx_zhi,
                               p_in, box_bounds):
    """Pure-numpy core of `energy_budget` — all inputs are already-read
    phasor line arrays (from real detectors or, in tests, synthetic ones),
    so this is unit-testable without running fdtdx at all.

    "port" fields (Ey_port, Hz_port, zs_port) are wg_mon's own (Ey, Hz) and
    z-sample grid; the four "ebox_*" arg pairs are the closure box's four
    faces in outward-face order (xlo, xhi, zlo, zhi) — zhi is always
    fiber_mon's own (Ey, Hx) (see `_box_bounds` / build_scene).
    """
    x_lo, x_hi, z_lo, z_hi = box_bounds
    dl = cfg.spacing_um

    Em, Hm_fwd, neff = slab_te0_mode(zs_port, 0.0, cfg)
    # judgment #5: there is no "transmitted past the grating, still
    # guided" channel in this scene to project onto here — wg_slab only
    # spans [-X0, -L_design/2] (build_scene: block("wg_slab", ...)), i.e.
    # the design region's LEFT edge. Everything to the right of the design
    # window (face_out_xhi below) is bare air/BOX/substrate: whatever power
    # goes that way is unguided radiation, not a mode to overlap against.
    # "forward" here means the -x direction (out of the box, into the real
    # output waveguide) matching `characterize`'s own P_mode convention.
    P_fwd = overlap_power_directional(Ey_port, Hz_port, Em, -Hm_fwd, dl)
    P_back = overlap_power_directional(Ey_port, Hz_port, Em, Hm_fwd, dl)

    # judgment #2: signed, not phasor_line_power's abs() — this quantity
    # must be able to cancel against the box faces below.
    port_face_net_in = -signed_poynting_flux_x(Ey_port, Hz_port, dl)
    if port_face_net_in <= 0:
        raise RuntimeError(
            f"port_face_net_in = {port_face_net_in:.6g} <= 0: net power is "
            f"not flowing from the grating into the output waveguide at "
            f"wg_mon. The interface contract asserts this must be > 0; "
            f"refusing to report a budget built on a port that isn't "
            f"actually receiving power (check excitation/teeth/monitor "
            f"placement before trusting anything else in this dict).")

    face_out_xlo = -signed_poynting_flux_x(Ey_xlo, Hz_xlo, dl)
    face_out_xhi = +signed_poynting_flux_x(Ey_xhi, Hz_xhi, dl)
    face_out_zlo = -signed_poynting_flux_z(Ey_zlo, Hx_zlo, dl)
    face_out_zhi = +signed_poynting_flux_z(Ey_zhi, Hx_zhi, dl)

    closure_sum_outward = (face_out_xlo + face_out_xhi +
                          face_out_zlo + face_out_zhi)
    closure_residual_frac_of_input = closure_sum_outward / port_face_net_in
    injection_purity_check = (port_face_net_in + P_back) / P_fwd

    return {
        "denominator": "P_fwd = forward slab-TE0 overlap at wg_mon",
        "P_fwd": float(P_fwd),
        "P_back": float(P_back),
        "n_eff": float(neff),
        "port_face_net_in": float(port_face_net_in),
        "face_out_xlo": float(face_out_xlo),
        "face_out_xhi": float(face_out_xhi),
        "face_out_zlo": float(face_out_zlo),
        "face_out_zhi": float(face_out_zhi),
        "closure_sum_outward": float(closure_sum_outward),
        "closure_residual_frac_of_input": float(closure_residual_frac_of_input),
        "injection_purity_check": float(injection_purity_check),
        "closure_check": check_energy_closure(
            closure_residual_frac_of_input,
            face_terms=(face_out_xlo, face_out_xhi,
                        face_out_zlo, face_out_zhi),
            port_face_net_in=port_face_net_in),
        "P_in": float(p_in),
        "CE": float(P_fwd / p_in),
        "CE_dB": float(10 * np.log10(P_fwd / p_in + 1e-15)),
        "box_bounds_um": {"x_lo": x_lo, "x_hi": x_hi,
                               "z_lo": z_lo, "z_hi": z_hi},
    }


def energy_budget(cfg, teeth, p_in=None, azimuth_sign=None, seed=0,
                  shallow_teeth=None):
    """Where did the power go? Fiber-side energy accounting, promoted from a
    one-off manual tally to a first-class, conservation-checked measurement.
    Returns a dict (see module section header and the five judgments below);
    `check_energy_closure` runs automatically and is attached under
    "closure_check".

    Five hard-won judgments this function encodes (do not undo any of
    them without re-reading why):

    1. PhasorDetector only, never fdtdx.PoyntingFluxDetector. The latter is
       an INSTANTANEOUS time-domain quantity; this scene runs a pulsed
       source and reports phasor (steady-state, single-frequency) CE, so
       mixing in a time-domain "flux" is a unit/physics error that produces
       a plausible-looking but meaningless number. wg_mon/fiber_mon already
       carry (Ey,Hz)/(Ey,Hx) phasors with full flux information — reuse
       those, plus three new PhasorDetector lines for the box (build_scene
       with_energy_box=True).
    2. `phasor_line_power`'s abs() throws away direction — fine for a
       normalization ratio (CE), wrong for an energy-conservation SUM.
       `signed_poynting_flux_x/z` keep the sign; their positive control
       (a synthetic pure +x TE0 mode measured by the mode-overlap formula
       AND the raw signed-flux formula must agree to ~1e-6, tests/
       test_energy_budget.py) proves both readers share one absolute power
       scale.
    3. The box in `_box_bounds` must exclude PML and any active source, or
       "four faces sum to zero" isn't true and the whole closure check is
       a category error. Violations RAISE (ValueError) rather than
       returning a number — see `_box_bounds`.
    4. `check_energy_closure` gates BOTH directions: too big (>0.5%, the
       books don't balance) AND too small (<0.1%, implausibly perfect for
       a finite pulsed run — see its docstring for why that's suspicious,
       not reassuring).
    5. There is no "transmitted past the grating, still guided" channel in
       this scene (wg_slab ends at the design window's left edge,
       build_scene:block("wg_slab", ...)); face_out_xhi is pure radiation
       by construction, not a mode to look for.
    """
    box_bounds = _box_bounds(cfg)   # raise before paying for a simulation

    slope = None
    if p_in is None or azimuth_sign is None:
        p_in, azimuth_sign, slope = calibrated_beam(cfg, seed=seed)

    arrays = _run(cfg, teeth=teeth, with_chip=True, seed=seed,
                  azimuth_sign=azimuth_sign, shallow_teeth=shallow_teeth,
                  with_energy_box=True)

    Ey_port = _phasor(arrays, "wg_mon", 0, y_axis=0)
    Hz_port = _phasor(arrays, "wg_mon", 1, y_axis=0)
    n = Ey_port.shape[0]
    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    zs_port = z_mon_lo + (np.arange(n) + 0.5) * cfg.spacing_um

    Ey_xlo = _phasor(arrays, "ebox_xlo", 0, y_axis=0)
    Hz_xlo = _phasor(arrays, "ebox_xlo", 1, y_axis=0)
    Ey_xhi = _phasor(arrays, "ebox_xhi", 0, y_axis=0)
    Hz_xhi = _phasor(arrays, "ebox_xhi", 1, y_axis=0)
    Ey_zlo = _phasor(arrays, "ebox_zlo", 0, y_axis=1)
    Hx_zlo = _phasor(arrays, "ebox_zlo", 1, y_axis=1)
    # face_out_zhi reuses fiber_mon directly (same plane, see _box_bounds)
    Ey_zhi = _phasor(arrays, "fiber_mon", 0, y_axis=1)
    Hx_zhi = _phasor(arrays, "fiber_mon", 1, y_axis=1)

    out = _energy_budget_from_fields(
        cfg, zs_port, Ey_port, Hz_port,
        Ey_xlo, Hz_xlo, Ey_xhi, Hz_xhi, Ey_zlo, Hx_zlo, Ey_zhi, Hx_zhi,
        p_in, box_bounds)
    out["azimuth_sign"] = azimuth_sign
    out["tilt_slope_rad_per_um"] = slope
    return out


# --------------------------------------------------------------------------
# 3D validation (extrude the 1D profile and re-measure it in 3D, on GPU)
# --------------------------------------------------------------------------


def build_scene_3d(cfg, teeth, wg_width_um=10.0, azimuth_sign=1.0,
                   with_chip=True, with_field_map=False):
    """3D scene: the quasi-2D layout extruded to a straight grating of width
    W (y), radial Gaussian beam from above, PML on all six sides.

    Same grating_coupler coordinates; y is now real (W + 2*1.5um margins + 2*dpml).
    """
    import jax.numpy as jnp
    from ..engines.fdtdx_fixes import GaussianBeamSource

    _require_float32_dtype(cfg)
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
        """Block in grating_coupler coords; full y unless y_width (centered) given."""
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
    src_width_y = cell_y - 2 * cfg.dpml - 1   # 3D beam aperture in y
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
                 "title": f"3D side view Re Ey  (y=0, λ = {cfg.lam_c} µm)"}
    out["xy"] = {"field": xy,
                 "eps": eps_grid_xy(cfg, teeth, wg_width_um, *xy.shape,
                                    cell_y_um=cell_y),
                 "extent": np.array((-cfg.X0, cfg.X0, -cell_y / 2,
                                     cell_y / 2)),
                 "title": f"3D top view Re Ey  (mid-silicon, "
                          f"λ = {cfg.lam_c} µm)"}
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
    """3D CE (fiber-side excitation, vertical/tilted radial beam) — the 3D
    twin of `characterize`. Two runs: empty-cell beam power + grating run.
    When lams_um is given, CE(lambda) comes from the same two runs
    (multi-wavelength phasors) — always prefer the spectrum, per conventions
    lesson 6.
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
    xs = (np.arange(nx) + 0.5) * cfg.spacing_um - cfg.X0   # grating_coupler x
    zs = (np.arange(nz) + 0.5) * cfg.spacing_um - cfg.Z0   # grating_coupler y
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
            "title": f"coupler steady-state field Re Ey  "
                     f"(λ = {cfg.lam_c} µm)"}


def gaussian_mode_tilted(xs_um, x0, w0, lam_um, theta_deg, kx_sign=-1.0):
    """Tilted upward Gaussian target on a horizontal line (complex fields).

    Eg = exp(-((x-x0)/w0)^2) * exp(i * kx_sign * k0 * sin(theta) * (x-x0)),
    Hg = -cos(theta) * Eg — the MINUS marks upward (+z) propagation: for
    E along y, an upward wave has Hx = -Ey while a downward one has +Ey
    (verified on the injected fiber-side beam). Getting this sign wrong makes
    the directional overlap reject the real signal entirely (~ -60 dB).
    kx_sign=-1 is the reciprocal partner of the incoming fiber-side beam
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
    """Waveguide-side excitation, ONE run -> (CE_fwd, S11, P_in).

    P_in  = forward TE0 overlap at wg_mon (filters injection impurity; the
            a_fwd normalization pattern)
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
    # fiber_mon is centered on the cell center = grating_coupler x=0, so these ARE grating_coupler
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
    then a LOWER BOUND, never silently exact: a reported number must not
    claim more than the sampling supports.
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
    empty-cell run (multi-wavelength phasors). The sparse-sampling lesson
    operationalized: dense spectra must be cheap or they get skipped.

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


# --------------------------------------------------------------------------
# Inverse design — the differentiable rho -> CE path
# --------------------------------------------------------------------------
#
# `profile_teeth` binarizes with `> 0.5` and run-length-encodes into
# UniformMaterialObject blocks: excellent for MEASURING a finished design,
# but a dead end for gradients. This section adds the second, differentiable
# rho -> permittivity route (one `fdtdx.Device` over the design window) and
# a jnp twin of the CE measurement chain, so `jax.value_and_grad` reaches
# every design voxel in one backward pass.
#
# The two routes must agree on binary designs — same grid, same tooth edges,
# only the permittivity WRITE differs (Device linearly interpolates inverse
# permittivity; UniformMaterialObject fills whole cells). That agreement is
# an acceptance criterion, not an assumption: compare `ce_from_arrays` and
# `characterize` on one rasterized grating before trusting any optimization.
#
# Grid constraints (hard, checked by callers): the Device z voxel is t_si
# snapped by round(t/resolution), so t_si/spacing_um must be an integer, and
# 1/design_grid_per_um must be an integer multiple of spacing_um. With the
# defaults (t_si = 0.220) only spacing_um in {0.020, 0.010} is clean; the
# module default 0.0125 snaps t_si to 0.225 um and must NOT be used here.


def design_device(cfg, name="design", with_transforms=True):
    """The one differentiable rho -> permittivity object: an air/Si Device
    covering the design window ([-L_design/2, +L_design/2] x full y x t_si)
    with one voxel per design pixel.

    with_transforms=True installs the production parameter chain
    (ConicFilter1D(radius = cfg.filter_radius) -> TanhProjection(eta_i)), so
    the latent parameters are filtered and projected before they become
    material indices. with_transforms=False makes the parameters the physical
    density itself — the only way to place a PRESCRIBED profile (e.g. a
    rasterized grating) into the Device path for comparison against the
    `characterize` measurement chain.
    """
    transforms = []
    if with_transforms:
        from ..fab.transforms import ConicFilter1D

        transforms = [ConicFilter1D(radius_um=cfg.filter_radius, axis=0),
                      fdtdx.TanhProjection(projection_midpoint=cfg.eta_i)]
    return fdtdx.Device(
        name=name,
        materials={"air": fdtdx.Material(permittivity=1.0),
                   "si": fdtdx.Material(permittivity=cfg.n_si ** 2)},
        param_transforms=transforms,
        partial_real_shape=(cfg.L_design * UM, None, cfg.t_si * UM),
        # one design pixel wide, the full thin y axis, the full Si thickness
        partial_voxel_real_shape=(UM / cfg.design_grid_per_um,
                                  cfg.n_y_cells * cfg.spacing_um * UM,
                                  cfg.t_si * UM),
    )


def n_design_voxels(cfg):
    """Length of the design vector — must match what script 07 expects
    (L_design is pinned at its default there, so never change it)."""
    return int(round(cfg.L_design * cfg.design_grid_per_um))


def assert_design_grid_snaps(cfg):
    """Fail loudly when the Device cannot be placed without snapping error.

    design_grid_per_um is annotated `int` on the config dataclass, but a
    dataclass annotation is not enforced at runtime -- a non-integer or
    non-positive value would otherwise pass through silently (a negative
    value in particular satisfies the divisibility check below by accident,
    since it only compares magnitudes). Reject it explicitly first: a
    non-integer or <=0 design_grid_per_um makes 1/design_grid_per_um (the
    design pixel size) and its alignment to the simulation grid undefined.
    """
    g = cfg.design_grid_per_um
    if g <= 0 or float(g) != int(g):
        raise ValueError(
            f"design_grid_per_um must be a positive integer (it is the "
            f"design-pixel density in pixels/um; 1/design_grid_per_um is "
            f"undefined otherwise) — got {g!r}.")
    for label, length in (("t_si", cfg.t_si),
                          ("design pixel", 1.0 / cfg.design_grid_per_um),
                          ("L_design", cfg.L_design)):
        n = length / cfg.spacing_um
        if abs(n - round(n)) > 1e-9:
            raise ValueError(
                f"{label} = {length} um is not an integer multiple of "
                f"spacing_um = {cfg.spacing_um} um ({n:.6f} cells): the "
                f"fdtdx Device would snap it and the design grid would no "
                f"longer line up with the measurement grid. Use "
                f"spacing_um = 0.020 (design_grid_per_um in 50/25/10) or "
                f"0.010 (100/50/25/20).")
    if 1.0 / cfg.spacing_um < cfg.design_grid_per_um - 1e-9:
        raise ValueError(
            f"1/spacing_um = {1 / cfg.spacing_um:.1f} < design_grid_per_um = "
            f"{cfg.design_grid_per_um}: adjoint gradients are systematically "
            f"underestimated below the design grid (conventions lesson 3).")


def build_scene_design(cfg, num_checkpoints=20, excitation="fiber",
                       lams_um=None, with_transforms=True,
                       gradient_config=None):
    """`build_scene` plus the design Device and a checkpointed GradientConfig.

    Everything that defines the physics — source, PML, monitors, stack — comes
    from the existing `build_scene(teeth=None, with_chip=True)`, so the
    differentiable FOM measures the same device as `characterize` does; only
    the grating itself is replaced by the Device.

    with_transforms=False makes the latent parameters the physical density
    (see `design_device`) — the acceptance-check path, not the design path.

    gradient_config=None (the default) keeps the historical behavior exactly:
    a checkpointed fdtdx.GradientConfig built from num_checkpoints. Passing a
    ready-made fdtdx.GradientConfig instead (e.g. method="reversible" with a
    Recorder) uses it verbatim and num_checkpoints is ignored — a
    pure opt-in switch; no existing caller passes it, so the checkpointed
    path is untouched.

    Returns (sim_config, object_list, constraints, device). The returned
    device is the UNPLACED template (use `objects.devices[0]` after
    `place_objects` for anything that evaluates the transform chain).
    """
    assert_design_grid_snaps(cfg)
    prev_lams = getattr(cfg, "_lams_um", None)
    if lams_um is not None:
        cfg._lams_um = tuple(lams_um)
    try:
        sim_config, object_list, constraints = build_scene(
            cfg, teeth=None, with_chip=True, excitation=excitation)
    finally:
        cfg._lams_um = prev_lams

    if gradient_config is None:
        gradient_config = fdtdx.GradientConfig(
            method="checkpointed", num_checkpoints=num_checkpoints)
    sim_config = sim_config.aset("gradient_config", gradient_config)

    device = design_device(cfg, with_transforms=with_transforms)
    volume = object_list[0]
    # `same_size(volume, axes=(1,))` below spans the Device across the whole y
    # axis. On the quasi-2D cell that is the intent -- y is n_y_cells of
    # periodic padding, so "the whole axis" is the extrusion.
    #
    # The harm on a cell with a real y axis is that the design is smeared
    # across it: every y row carries the same density, so the optimiser has no
    # y freedom while the array claims to. It is NOT "silicon inside the y
    # PML" -- an earlier version of this comment said that, and it is false
    # here: build_scene sets boundary_type_miny/maxy = "periodic" (:394), so
    # this path has no y PML at all. (build_scene_3d is different: it takes
    # BoundaryConfig.from_uniform_bound, whose y default IS pml, which is why
    # the comment there is correct.)
    # `build_scene_design_3d` uses place_at_center with an explicit L_design_y
    # for exactly this reason; this check makes the assumption load-bearing
    # rather than a comment someone can walk past.
    # Read the marker off the built scene, not off cfg. An earlier version
    # compared volume.partial_real_shape[1] against n_y_cells*spacing_um --
    # the very expression build_scene had just used to construct it, so the
    # difference was identically zero and the guard could never fire. Measured
    # at n_y_cells=300 (a 3.75 um y axis, exactly the dangerous case): diff
    # 0.00e+00, passed silently. A guard fed by the knob it is checking is not
    # a guard.
    #
    # The structural difference between the two builders is the y boundary:
    # build_scene sets miny/maxy periodic and emits two BlochBoundary objects;
    # build_scene_3d takes BoundaryConfig.from_uniform_bound and emits six
    # PMLs with no Bloch at all.
    n_bloch = sum(1 for o in object_list
                  if type(o).__name__ == "BlochBoundary")
    if n_bloch != 2:
        raise ValueError(
            f"build_scene_design spans the Device across the whole y axis, "
            f"which is only correct when y is periodic padding. This scene "
            f"has {n_bloch} Bloch boundaries (the quasi-2D cell has 2); its y "
            f"is a real axis, and spanning it smears one design across every "
            f"y row. Use build_scene_design_3d.")
    # same anchoring as build_scene's block(): grating_coupler coords -> scene coords
    constraints.append(device.place_relative_to(
        volume, axes=(0, 2), own_positions=(-1, -1), other_positions=(-1, -1),
        margins=((cfg.X0 - cfg.L_design / 2) * UM, cfg.Z0 * UM)))
    constraints.append(device.same_size(volume, axes=(1,)))
    object_list.append(device)
    return sim_config, object_list, constraints, device


def te0_target_on_monitor(cfg, nz_mon, lam_um=None):
    """Static (numpy) -x traveling TE0 target on the wg_mon line.

    Returns (Em, Hm_back, Pm, n_eff) with Pm the mode power that
    `overlap_power_directional` divides by — computed here once so the
    traced FOM never re-solves the analytic mode.
    """
    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    zs = z_mon_lo + (np.arange(nz_mon) + 0.5) * cfg.spacing_um
    mode_cfg = cfg
    if lam_um is not None and abs(float(lam_um) - cfg.lam_c) > 1e-12:
        # only the lam-dependent mode parameters matter (characterize_spectrum)
        mode_cfg = type(cfg)()
        mode_cfg.lam_c, mode_cfg.t_si = float(lam_um), cfg.t_si
        mode_cfg.n_si, mode_cfg.n_sio2 = cfg.n_si, cfg.n_sio2
    Em, Hm_fwd, neff = slab_te0_mode(zs, 0.0, mode_cfg)
    Hm_back = -Hm_fwd
    Pm = abs(0.5 * np.real(np.sum(Em * np.conj(Hm_back))) * cfg.spacing_um)
    return Em, Hm_back, float(Pm), float(neff)


def overlap_power_directional_jnp(E, H, Em, Hm, dl, Pm):
    """jnp twin of `overlap_power_directional` with Pm precomputed.

    Same 1/2 power convention, same 1/4 coupling coefficient; the numpy
    original recomputes Pm from (Em, Hm) every call, this one takes it as a
    static number so the traced graph stays tiny. Parity with the numpy
    version is a unit test (tests/test_grating_coupler_optimize.py), not a claim.
    """
    import jax.numpy as jnp

    Emj = jnp.asarray(Em)
    Hmj = jnp.asarray(Hm)
    a = 0.25 * jnp.sum(E * jnp.conj(Hmj) + jnp.conj(Emj) * H) * dl
    return jnp.abs(a) ** 2 / Pm


def ce_from_arrays(arrays, cfg, target, p_in, lam_idx=0):
    """Differentiable twin of `characterize`'s CE, read off a finished run.

    arrays — the ArrayContainer returned by a differentiable run of the
             `build_scene_design` scene (wg_mon PhasorDetector present)
    target — `te0_target_on_monitor(...)` for the same lam_idx
    p_in   — incident beam power from the empty-cell run at that wavelength
             (a plain float: it does not depend on the design)
    """
    import jax.numpy as jnp

    Em, Hm, Pm, _ = target
    ph = arrays.detector_states["wg_mon"]["phasor"]
    # wg_mon is an x-plane: (nt=1, n_lam, n_comp, 1, ny, nz); average the thin
    # periodic y axis exactly as the numpy `_phasor` reader does.
    #
    # That average IS the extrusion assumption. It is correct here and only
    # here: this FOM belongs to the quasi-2D scene, whose y axis is n_y_cells
    # of periodic padding carrying no design freedom. Point it at a scene with
    # a real y axis -- `build_scene_design_3d`, where the Device has design
    # pixels in y -- and the optimiser is handed a y-averaged objective. The
    # run exits 0, the numbers look ordinary, and they mean nothing. Nothing
    # downstream can tell, which is why the check is here and not in a test.
    ny_seen = ph.shape[-2]
    if ny_seen != cfg.n_y_cells:
        raise ValueError(
            f"ce_from_arrays averages the y axis away, but wg_mon has "
            f"ny={ny_seen} against cfg.n_y_cells={cfg.n_y_cells}: this is not "
            f"the quasi-2D scene. Averaging a real y axis silently applies the "
            f"extrusion assumption where it does not hold. Use "
            f"ce_from_arrays_3d (full (y,z) plane overlap) for that scene.")
    Ey = jnp.mean(ph[0, lam_idx, 0], axis=(0, 1))
    Hz = jnp.mean(ph[0, lam_idx, 1], axis=(0, 1))
    p_mode = overlap_power_directional_jnp(Ey, Hz, Em, Hm,
                                           cfg.spacing_um, Pm)
    return p_mode / p_in


def beam_power_spectrum(cfg, lams_um, seed=0, azimuth_sign=1.0):
    """Incident beam power per wavelength from ONE empty-cell run.

    The multi-wavelength companion of `beam_power_and_tilt` (which only
    reports lam_c): needed because a multi-wavelength FOM must normalize each
    wavelength by its own P_in. Calibrate the tilt with `calibrated_beam`
    first — the slope check is only valid at lam_c.
    """
    prev_lams = getattr(cfg, "_lams_um", None)
    cfg._lams_um = tuple(lams_um)
    try:
        arrays = _run(cfg, teeth=None, with_chip=False, seed=seed,
                      azimuth_sign=azimuth_sign)
    finally:
        cfg._lams_um = prev_lams
    out = []
    for k in range(len(tuple(lams_um))):
        Ey = _phasor(arrays, "fiber_mon", 0, y_axis=1, lam_idx=k)
        Hx = _phasor(arrays, "fiber_mon", 1, y_axis=1, lam_idx=k)
        out.append(phasor_line_power(Ey, Hx, cfg.spacing_um))
    return out


def make_ce_value_and_grad(cfg, p_in, num_checkpoints=20, lams=None,
                           with_transforms=True, seed=None,
                           gradient_config=None,
                           expected_feature_nm=23.0):
    """Build the differentiable grating_coupler FOM once; return the compiled callables.

    Returns (vg_fn, objects, arrays, params0, device, value_fn) where

        vg_fn(p, beta)    -> (loss, dloss/dp),  loss = -CE  (minimize)
        value_fn(p, beta) -> loss                (finite differences)

    `p` is the latent design array of shape (n_design_voxels, 1, 1) and
    `beta` is passed as a TRACED argument, so the whole beta schedule runs on
    a single compilation (fdtdx's TanhProjection takes beta as a kwarg of
    apply_params).

    Multiple `lams` are aggregated with the smooth minimum (cfg.softmin_beta)
    — worst-wavelength-first, at essentially zero extra cost because one run
    produces every phasor. `p_in` is then a per-wavelength sequence.

    cfg.w_s11 > 0 switches to ONE waveguide-side excitation per
    evaluation: FOM = CE - w_s11*R11 with the CE term the reciprocal upward
    Gaussian overlap and R11 the backward TE0 overlap, both normalized by
    the TRACED forward overlap (`p_in` is then unused). vg_fn returns
    ((loss, {"ce", "s11"}), grad) — loss = -FOM, aux holds the true CE and
    the linear R11; value_fn still returns the scalar loss.

    `device` is the PLACED device (its transform chain is initialized), which
    is what `rho_from_params` needs.

    gradient_config — forwarded to `build_scene_design`; None (default) keeps
    the checkpointed path exactly as before, a non-None value replaces it
    verbatim (num_checkpoints is then ignored). See `build_scene_design`.
    """
    import jax.numpy as jnp

    from ..fab.filters_jax import softmin

    lams = tuple(lams) if lams else (cfg.lam_c,)
    # Lesson 5 (engines/conventions.py): a minimax FOM over sparse wavelength
    # samples lets the optimiser pump the sampled points while the spectrum
    # collapses between them -- an earlier prototype did exactly that, looking
    # healthy on 3 samples spread over 100 nm while a deep valley opened
    # between two of them. The guard existed and had zero call sites; this is
    # the throat both drivers pass through.
    #
    # The default feature width is 2*gamma = 23.0 nm, from a two-grid
    # same-window fit of the resonance notch. It is a measurement INSIDE the
    # fdtdx representation. Another solver's convergence trend extrapolates
    # to a smaller gamma, but that is a number in its representation; this
    # guard governs the spacing of this FOM's own samples, so it uses this
    # ruler. An earlier draft used 22.08 nm from an older fit window -- no
    # material difference, but the value is now pinned to the two-grid
    # same-window fit rather than to whichever fit happened to be at hand.
    #
    # Not a general constant: a narrower resonance needs a smaller value
    # passed in.
    if len(lams) > 1:
        from ..engines.conventions import assert_fom_sampling_covers_band
        # The WIDEST gap decides, not the narrowest: with min(), any two
        # samples placed close together let an arbitrarily large hole through,
        # and "dense in the middle, sparse at the edges" is the natural way to
        # write a --lams list. Measured: [1.3000, 1.3005, 1.4000] passed with a
        # 99.5 nm gap. That gap at the band edge is exactly the pathology this
        # guard exists to stop.
        spacing_nm = max(abs(b - a) for a, b in zip(sorted(lams)[:-1],
                                                    sorted(lams)[1:])) * 1e3
        assert_fom_sampling_covers_band(spacing_nm, expected_feature_nm)
    p_in_list = ([float(p_in)] * len(lams) if np.isscalar(p_in)
                 else [float(v) for v in p_in])
    if len(p_in_list) != len(lams):
        raise ValueError(f"p_in has {len(p_in_list)} entries but "
                         f"{len(lams)} wavelengths were requested")

    use_wg = float(cfg.w_s11) > 0.0
    sim_config, objs, cons, template = build_scene_design(
        cfg, num_checkpoints=num_checkpoints,
        excitation="wg" if use_wg else "fiber",   # "fiber" == legacy default
        lams_um=lams if len(lams) > 1 else None,
        with_transforms=with_transforms,
        gradient_config=gradient_config)

    key = jax.random.PRNGKey(cfg.seed if seed is None else seed)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    device = next(d for d in objects.devices if d.name == template.name)

    nz_mon = arrays.detector_states["wg_mon"]["phasor"].shape[-1]
    targets = [te0_target_on_monitor(cfg, nz_mon, lam_um=l) for l in lams]

    if use_wg:
        w = float(cfg.w_s11)
        ph_f = arrays.detector_states["fiber_mon"]["phasor"]
        assert ph_f.shape[-1] == 1    # z-plane: (nt, n_lam, n_comp, nx, ny, 1)
        nf = ph_f.shape[3]
        # fiber_mon is centered on the cell center = grating_coupler x=0 (see
        # wg_side_characterize); the target is complex when tilted and the
        # phasor time-reference varies between run configurations, so both
        # kx signs are built and the traced CE takes the larger overlap
        xs = (np.arange(nf) - nf / 2) * cfg.spacing_um
        gtargets = []
        for l in lams:
            per = []
            for kx in (-1.0, +1.0):
                Eg, Hg = gaussian_mode_tilted(xs, cfg.fiber_x0, cfg.w0, l,
                                              cfg.theta_deg, kx_sign=kx)
                Pg = abs(0.5 * np.real(np.sum(Eg * np.conj(Hg)))
                         * cfg.spacing_um)
                per.append((Eg, Hg, float(Pg)))
            gtargets.append(per)

        def loss(p, beta):
            prm = dict(params)
            prm[device.name] = p
            a, o, _ = fdtdx.apply_params(arrays, objects, prm, k2, beta=beta)
            _, a = run_fdtd_buffers(arrays=a, objects=o, config=sim_config,
                                     key=key, show_progress=False)   # ONE sim
            ces, r11s = [], []
            for k in range(len(lams)):
                # te0_target_on_monitor returns the -x mode (Hm_back), so
                # the injected +x forward overlap flips its sign
                Em, Hm_back, Pm, _ = targets[k]
                ph = a.detector_states["wg_mon"]["phasor"]
                Ey = jnp.mean(ph[0, k, 0], axis=(0, 1))   # as ce_from_arrays
                Hz = jnp.mean(ph[0, k, 1], axis=(0, 1))
                p_fwd = overlap_power_directional_jnp(
                    Ey, Hz, Em, -Hm_back, cfg.spacing_um, Pm)
                p_back = overlap_power_directional_jnp(
                    Ey, Hz, Em, Hm_back, cfg.spacing_um, Pm)
                phf = a.detector_states["fiber_mon"]["phasor"]
                Eyf = jnp.mean(phf[0, k, 0], axis=(1, 2))
                Hxf = jnp.mean(phf[0, k, 1], axis=(1, 2))
                pg = jnp.maximum(*[overlap_power_directional_jnp(
                    Eyf, Hxf, Eg, Hg, cfg.spacing_um, Pg)
                    for Eg, Hg, Pg in gtargets[k]])
                ces.append(pg / p_fwd)     # traced P_in — NOT static p_in_list
                r11s.append(p_back / p_fwd)
            foms = [ces[k] - w * r11s[k] for k in range(len(lams))]
            fom = foms[0] if len(foms) == 1 else softmin(jnp.stack(foms),
                                                         cfg.softmin_beta)
            ce = ces[0] if len(ces) == 1 else softmin(jnp.stack(ces),
                                                      cfg.softmin_beta)
            r11 = r11s[0] if len(r11s) == 1 else jnp.max(jnp.stack(r11s))
            return -fom, {"ce": ce, "s11": r11}

        return (jax.jit(jax.value_and_grad(loss, has_aux=True)), objects,
                arrays, params, device,
                jax.jit(lambda p, beta: loss(p, beta)[0]))  # scalar for FD

    def loss(p, beta):
        prm = dict(params)
        prm[device.name] = p
        a, o, _ = fdtdx.apply_params(arrays, objects, prm, k2, beta=beta)
        _, a = run_fdtd_buffers(arrays=a, objects=o, config=sim_config, key=key,
                                 show_progress=False)
        ces = [ce_from_arrays(a, cfg, targets[k], p_in_list[k], lam_idx=k)
               for k in range(len(lams))]
        ce = ces[0] if len(ces) == 1 else softmin(jnp.stack(ces),
                                                  cfg.softmin_beta)
        return -ce

    return (jax.jit(jax.value_and_grad(loss)), objects, arrays, params,
            device, jax.jit(loss))


def rho_from_params(device, params, beta):
    """Host-side physical density (0..1, length n_design_voxels) of a latent
    parameter vector — the transform chain evaluated outside the simulation.

    This is what goes into design_rho_cont.npy; its `> 0.5` binarization is
    design_rho.npy, the file script 07 re-measures.
    """
    import jax.numpy as jnp

    p = params[device.name] if isinstance(params, dict) else params
    dens = device(jnp.asarray(p), expand_to_sim_grid=False,
                  beta=jnp.asarray(beta, dtype=jnp.float32))
    return np.asarray(dens, dtype=float).reshape(-1)


def rasterize_teeth(cfg, teeth):
    """Inverse of `profile_teeth`: teeth (grating_coupler x-coords) -> design vector.

    Each tooth is rounded to whole pixels the way fdtdx rounds a placed
    UniformMaterialObject — nearest pixel for the left edge, nearest pixel
    count for the width — so an off-grid grating renders here to the same
    device fdtdx would build from the teeth directly. That equivalence is the
    whole point: `--init grating` is the only starting design in this repo
    with a cross-validated baseline, and it only IS that design if the
    rasterization preserves it.

    A pixel-centre rule (silicon where the pixel centre falls inside a tooth)
    looks equally defensible and is not: it lets the tooth WIDTH alternate
    between 14 and 15 pixels across a 0.575 um / 20 nm grating, and the
    resulting +-3.5% duty jitter moves the coupling ridge off the design
    wavelength far enough to wreck the coupling efficiency there. That is
    conventions lesson 6 biting inside a single engine.

    Round-tripping a grid-aligned profile is exact either way, which is what
    the unit test pins.
    """
    n = n_design_voxels(cfg)
    grid = cfg.design_grid_per_um
    rho = np.zeros(n)
    for x_min, w in (teeth or []):
        i0 = int(round((x_min + cfg.L_design / 2) * grid))
        i1 = i0 + int(round(w * grid))
        rho[max(i0, 0):max(min(i1, n), 0)] = 1.0
    return rho


# --------------------------------------------------------------------------
# 2D free-form inverse design — the differentiable xi(x,y) -> CE path in a
# REAL 3D scene
# --------------------------------------------------------------------------
#
# Everything above this line is the 1D path (xi(x) extruded along a thin
# periodic y) and stays byte-identical — earlier results depend on it. This
# section is the additive second path: a (nx, ny) design Device inside
# `build_scene_3d`'s geometry (true y extent, PML on all six sides), filtered
# by a conic filter whose kernel is the Euclidean distance in the (x, y)
# plane (ConicFilter2D), measured by the separable 3D waveguide-mode overlap
# on the wg_mon PLANE (no y averaging — the y average at `ce_from_arrays` IS
# the extrusion assumption, and it is the single most dangerous line on this
# path).
#
# Scope of this path: CE-only (w_s11 == 0, static empty-cell P_in exactly
# like the legacy CE-only FOM); no 2D min-feature measurement (fab/measure.py
# is 1D-only by its own docstring), no GDS export, no teeth-based
# re-measurement (a 2D pattern has no teeth representation).


def design_shape_2d(cfg):
    """(nx, ny) of the 2D design array — the rank-2 twin of
    `n_design_voxels` (which stays scalar for the 1D path)."""
    return (int(round(cfg.L_design * cfg.design_grid_per_um)),
            int(round(cfg.L_design_y * cfg.design_grid_per_um)))


def assert_design_grid_snaps_2d(cfg, allow_t_si_snap=False):
    """The 2D twin of `assert_design_grid_snaps`: same x legs, plus the y
    legs (L_design_y and the y design pixel share the x pixel size, so the
    new leg is L_design_y alone), plus the same 1/spacing >= design grid
    guard. A separate function so the 1D guard stays byte-identical.

    allow_t_si_snap=True downgrades ONLY the t_si leg to a loud print: a
    coarse-grid chain-verification run (e.g. spacing 0.10 um) cannot
    satisfy t_si/spacing integer at all (0.220/0.10 = 2.2) and knowingly
    accepts the fdtdx round() snap of the silicon thickness — acceptable for
    "does the chain produce finite gradients and falling loss", never for a
    design deliverable. The design-pixel/L_design/L_design_y legs are NOT
    relaxable: those misalign the design grid against the measurement grid,
    which is a different and always-fatal error.
    """
    g = cfg.design_grid_per_um
    if g <= 0 or float(g) != int(g):
        raise ValueError(
            f"design_grid_per_um must be a positive integer (it is the "
            f"design-pixel density in pixels/um; 1/design_grid_per_um is "
            f"undefined otherwise) — got {g!r}.")
    for label, length, relaxable in (
            ("t_si", cfg.t_si, True),
            ("design pixel", 1.0 / cfg.design_grid_per_um, False),
            ("L_design", cfg.L_design, False),
            ("L_design_y", cfg.L_design_y, False)):
        n = length / cfg.spacing_um
        if abs(n - round(n)) > 1e-9:
            if relaxable and allow_t_si_snap:
                snapped = round(n) * cfg.spacing_um
                print(f"[grid] WARNING: {label} = {length} um is not an "
                      f"integer multiple of spacing_um = {cfg.spacing_um} um "
                      f"({n:.6f} cells); fdtdx will snap it to {snapped:.6g} "
                      f"um. Accepted because allow_t_si_snap=True — chain "
                      f"verification only, NOT a design deliverable.")
                continue
            raise ValueError(
                f"{label} = {length} um is not an integer multiple of "
                f"spacing_um = {cfg.spacing_um} um ({n:.6f} cells): the "
                f"fdtdx Device would snap it and the design grid would no "
                f"longer line up with the measurement grid.")
    if 1.0 / cfg.spacing_um < cfg.design_grid_per_um - 1e-9:
        raise ValueError(
            f"1/spacing_um = {1 / cfg.spacing_um:.1f} < design_grid_per_um = "
            f"{cfg.design_grid_per_um}: adjoint gradients are systematically "
            f"underestimated below the design grid (conventions lesson 3).")


def design_device_2d(cfg, name="design", with_transforms=True):
    """The 2D free-form rho(x, y) -> permittivity Device: air/Si over the
    design window ([-L_design/2, +L_design/2] x [-L_design_y/2, +L_design_y/2]
    x [0, t_si]) with one voxel per (x, y) design pixel — the rank-2 twin of
    `design_device`, which extrudes one y voxel across the whole cell.

    with_transforms=True installs ConicFilter2D(radius = cfg.filter_radius)
    -> TanhProjection(eta_i); with_transforms=False makes the parameters the
    physical density itself (prescribed-profile placement, the acceptance-
    check path).
    """
    transforms = []
    if with_transforms:
        from ..fab.transforms import ConicFilter2D

        transforms = [ConicFilter2D(radius_um=cfg.filter_radius, axes=(0, 1)),
                      fdtdx.TanhProjection(projection_midpoint=cfg.eta_i)]
    pixel = UM / cfg.design_grid_per_um
    return fdtdx.Device(
        name=name,
        materials={"air": fdtdx.Material(permittivity=1.0),
                   "si": fdtdx.Material(permittivity=cfg.n_si ** 2)},
        param_transforms=transforms,
        partial_real_shape=(cfg.L_design * UM, cfg.L_design_y * UM,
                            cfg.t_si * UM),
        # one design pixel in x AND y, the full Si thickness in z
        partial_voxel_real_shape=(pixel, pixel, cfg.t_si * UM),
    )


def build_scene_design_3d(cfg, num_checkpoints=20, lams_um=None,
                          with_transforms=True, wg_width_um=10.0,
                          gradient_config=None, allow_t_si_snap=False):
    """`build_scene_3d` plus the 2D design Device and a GradientConfig — the
    differentiable 3D scene.

    Geometry, source, monitors and PML all come from the existing
    `build_scene_3d(teeth=None, with_chip=True)` — real y extent, radial
    beam, PML on all six sides — so the differentiable FOM measures the same
    scene `characterize_3d` does; only the grating teeth are replaced by the
    Device. A third builder on purpose: `build_scene` and `build_scene_3d`
    stay byte-identical so every previously stored result stays reproducible.

    gradient_config=None builds the checkpointed fdtdx.GradientConfig from
    num_checkpoints (the default, characterized path); a ready-made
    GradientConfig (e.g. reversible) is used verbatim, exactly like
    `build_scene_design`.

    Returns (sim_config, object_list, constraints, device); the device is the
    UNPLACED template.
    """
    assert_design_grid_snaps_2d(cfg, allow_t_si_snap=allow_t_si_snap)
    prev_lams = getattr(cfg, "_lams_um", None)
    if lams_um is not None:
        cfg._lams_um = tuple(lams_um)
    try:
        sim_config, object_list, constraints = build_scene_3d(
            cfg, teeth=None, wg_width_um=wg_width_um, with_chip=True)
    finally:
        cfg._lams_um = prev_lams

    if gradient_config is None:
        gradient_config = fdtdx.GradientConfig(
            method="checkpointed", num_checkpoints=num_checkpoints)
    sim_config = sim_config.aset("gradient_config", gradient_config)

    device = design_device_2d(cfg, with_transforms=with_transforms)
    volume = object_list[0]
    # x/z anchoring exactly as build_scene_design; y centered with its own
    # L_design_y extent (same_size on y would drive design Si into the y PML)
    constraints.append(device.place_relative_to(
        volume, axes=(0, 2), own_positions=(-1, -1), other_positions=(-1, -1),
        margins=((cfg.X0 - cfg.L_design / 2) * UM, cfg.Z0 * UM)))
    constraints.append(device.place_at_center(volume, axes=(1,)))
    object_list.append(device)
    return sim_config, object_list, constraints, device


def wg_mode_target_on_monitor_3d(cfg, ny_mon, nz_mon, wg_width_um=10.0,
                                 lam_um=None):
    """Static (numpy) -x traveling separable waveguide mode on the wg_mon
    PLANE — the rank-2 twin of `te0_target_on_monitor`.

    Same mode, same sampling and same area element as the numpy measurement
    chain (`wg_mode_3d` / `combine_3d`), computed once at trace time so the
    traced FOM never re-solves the mode. Returns (Em, Hm_back, Pm, n_eff)
    with Em/Hm_back of shape (ny_mon, nz_mon) and Pm the mode power on the
    plane (dA = spacing_um**2). The separable TE0(z)*cos(y) target is
    documented as good to a few percent for REPORTING; as an optimisation
    target that error is exploitable by the optimiser — a known limitation
    of this path, to revisit before any design-deliverable claim.
    """
    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    zs = z_mon_lo + (np.arange(nz_mon) + 0.5) * cfg.spacing_um
    ys = (np.arange(ny_mon) - ny_mon / 2 + 0.5) * cfg.spacing_um
    mode_cfg = cfg
    if lam_um is not None and abs(float(lam_um) - cfg.lam_c) > 1e-12:
        mode_cfg = type(cfg)()
        mode_cfg.lam_c, mode_cfg.t_si = float(lam_um), cfg.t_si
        mode_cfg.n_si, mode_cfg.n_sio2 = cfg.n_si, cfg.n_sio2
    Em, Hm_fwd, neff = wg_mode_3d(ys, zs, mode_cfg, wg_width_um)
    Hm_back = -Hm_fwd
    dA = cfg.spacing_um ** 2
    Pm = abs(0.5 * np.real(np.sum(Em * np.conj(Hm_back))) * dA)
    return Em, Hm_back, float(Pm), float(neff)


def ce_from_arrays_3d(arrays, cfg, target, p_in, lam_idx=0):
    """Differentiable twin of `combine_3d`'s CE, read off a finished 3D run.

    The overlap integral runs over the FULL (ny, nz) wg_mon plane with the
    area element dA = spacing_um**2 — deliberately NO y averaging (the
    `jnp.mean(..., axis=(0, 1))` of the quasi-2D `ce_from_arrays` is the
    extrusion assumption; kept there, a 2D design would be optimised against
    a y-averaged objective and the run would mean nothing).

    target — `wg_mode_target_on_monitor_3d(...)` for the same lam_idx
    p_in   — incident beam power from the empty-cell 3D run (plain float,
             design-independent, same `abs(0.5 Re sum(Ey Hx*)) dA` reading
             as `combine_3d`)
    """
    Em, Hm, Pm, _ = target
    ph = arrays.detector_states["wg_mon"]["phasor"]
    # wg_mon is an x-plane: (nt=1, n_lam, n_comp, 1, ny, nz); take the single
    # x slice and keep the whole (y, z) plane
    Ey = ph[0, lam_idx, 0, 0]
    Hz = ph[0, lam_idx, 1, 0]
    dA = cfg.spacing_um ** 2
    p_mode = overlap_power_directional_jnp(Ey, Hz, Em, Hm, dA, Pm)
    return p_mode / p_in


def beam_power_3d(cfg, lams_um, wg_width_um=10.0, seed=0):
    """Incident beam power per wavelength from ONE empty-cell 3D run — the
    3D twin of `beam_power_spectrum`, with `combine_3d`'s exact P_in reading
    (fiber_mon plane, dA area element)."""
    lams = tuple(lams_um) if lams_um else (cfg.lam_c,)
    planes = run_3d_planes(cfg, None, wg_width_um, False, lams, seed=seed)
    dA = cfg.spacing_um ** 2
    out = []
    for k in range(len(lams)):
        Ey = np.squeeze(planes["fiber"][0, k, 0])
        Hx = np.squeeze(planes["fiber"][0, k, 1])
        out.append(float(abs(0.5 * np.real(np.sum(Ey * np.conj(Hx))) * dA)))
    return out


def make_ce_value_and_grad_3d(cfg, p_in, num_checkpoints=20, lams=None,
                              with_transforms=True, seed=None,
                              wg_width_um=10.0, gradient_config=None,
                              allow_t_si_snap=False,
                              expected_feature_nm=23.0):
    """Build the differentiable 2D-free-form grating_coupler FOM once — the 3D twin of
    `make_ce_value_and_grad`, CE-only.

    Returns (vg_fn, objects, arrays, params0, device, value_fn) with the same
    contract: vg_fn(p, beta) -> (loss, dloss/dp), loss = -CE, beta traced.
    `p` is the latent design array of shape (nx, ny, 1) from
    `design_shape_2d`. Multiple `lams` aggregate with the smooth minimum;
    `p_in` is the static per-wavelength empty-cell power (`beam_power_3d`).

    w_s11 > 0 is NOT implemented on this path: the S11-penalised FOM needs a
    traced P_in on a 3D waveguide-side excitation, which this path does not
    build — out of scope here. Raising instead of silently optimising the
    wrong FOM.
    """
    import jax.numpy as jnp

    from ..fab.filters_jax import softmin

    if float(cfg.w_s11) > 0.0:
        raise NotImplementedError(
            f"w_s11 = {cfg.w_s11} > 0: the S11-penalised FOM is not "
            f"implemented on the 3D 2D-free-form path (needs a traced P_in "
            f"on a 3D wg-side excitation — this path is CE-only). Set "
            f"w_s11=0.")

    lams = tuple(lams) if lams else (cfg.lam_c,)
    # Lesson 5 (engines/conventions.py): a minimax FOM over sparse wavelength
    # samples lets the optimiser pump the sampled points while the spectrum
    # collapses between them -- an earlier prototype did exactly that, looking
    # healthy on 3 samples spread over 100 nm while a deep valley opened
    # between two of them. The guard existed and had zero call sites; this is
    # the throat both drivers pass through.
    #
    # The default feature width is the measured 3 dB width of the resonance
    # notch in the fdtdx representation (2*gamma = 22.08 nm), NOT a general
    # constant -- a narrower resonance needs a smaller value passed in, and
    # solvers do not yet agree on gamma, so this number is not settled.
    if len(lams) > 1:
        from ..engines.conventions import assert_fom_sampling_covers_band
        # The WIDEST gap decides, not the narrowest: with min(), any two
        # samples placed close together let an arbitrarily large hole through,
        # and "dense in the middle, sparse at the edges" is the natural way to
        # write a --lams list. Measured: [1.3000, 1.3005, 1.4000] passed with a
        # 99.5 nm gap. That gap at the band edge is exactly the pathology this
        # guard exists to stop.
        spacing_nm = max(abs(b - a) for a, b in zip(sorted(lams)[:-1],
                                                    sorted(lams)[1:])) * 1e3
        assert_fom_sampling_covers_band(spacing_nm, expected_feature_nm)
    p_in_list = ([float(p_in)] * len(lams) if np.isscalar(p_in)
                 else [float(v) for v in p_in])
    if len(p_in_list) != len(lams):
        raise ValueError(f"p_in has {len(p_in_list)} entries but "
                         f"{len(lams)} wavelengths were requested")

    sim_config, objs, cons, template = build_scene_design_3d(
        cfg, num_checkpoints=num_checkpoints,
        lams_um=lams if len(lams) > 1 else None,
        with_transforms=with_transforms, wg_width_um=wg_width_um,
        gradient_config=gradient_config,
        allow_t_si_snap=allow_t_si_snap)

    key = jax.random.PRNGKey(cfg.seed if seed is None else seed)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    device = next(d for d in objects.devices if d.name == template.name)

    ph_shape = arrays.detector_states["wg_mon"]["phasor"].shape
    ny_mon, nz_mon = int(ph_shape[-2]), int(ph_shape[-1])
    targets = [wg_mode_target_on_monitor_3d(cfg, ny_mon, nz_mon,
                                            wg_width_um, lam_um=l)
               for l in lams]

    def loss(p, beta):
        prm = dict(params)
        prm[device.name] = p
        a, o, _ = fdtdx.apply_params(arrays, objects, prm, k2, beta=beta)
        _, a = run_fdtd_buffers(arrays=a, objects=o, config=sim_config,
                                key=key, show_progress=False)
        ces = [ce_from_arrays_3d(a, cfg, targets[k], p_in_list[k], lam_idx=k)
               for k in range(len(lams))]
        ce = ces[0] if len(ces) == 1 else softmin(jnp.stack(ces),
                                                  cfg.softmin_beta)
        return -ce

    return (jax.jit(jax.value_and_grad(loss)), objects, arrays, params,
            device, jax.jit(loss))


def rho_from_params_2d(device, params, beta):
    """Host-side physical density rho(x, y) of a latent parameter array —
    the rank-2 twin of `rho_from_params` (which flattens to 1D and stays the
    1D path's contract). Returns shape (nx, ny); its `> 0.5` binarization is
    design_rho_2d.npy."""
    import jax.numpy as jnp

    p = params[device.name] if isinstance(params, dict) else params
    dens = device(jnp.asarray(p), expand_to_sim_grid=False,
                  beta=jnp.asarray(beta, dtype=jnp.float32))
    dens = np.asarray(dens, dtype=float)
    return dens.reshape(dens.shape[0], dens.shape[1])


# --------------------------------------------------------------------------
# Problem contract — the declaration the gates read (problems/contract.py)
# --------------------------------------------------------------------------


def gradcheck_case():
    """G2 Part C's case: finite-difference the production inverse-design path.

    Cheap-but-real settings: the full coupler scene at the 20 nm grid (the
    grid the production recipe uses, and the only one that divides t_si
    exactly), a
    0.15 ps run and 10 checkpoints. The FOM is the unnormalized mode power
    (P_in = 1): the incident-beam normalization is a design-independent
    constant, so leaving it out saves an empty-cell run without weakening the
    check by anything.

    The base design is the uniform grating softened to 0.1/0.9 — a mid-grey
    slab has almost no gradient signal to check (it is not a grating), and a
    hard 0/1 profile sits on the clip boundary where the central difference
    would degenerate into a one-sided one.

    Everything above is a property of THIS problem, which is why it lives
    here and not in the gate. What stays in the gate is the check itself: the
    eligibility floor (MIN_REL_GRAD), the voxel sampling, the Richardson
    extrapolation and REL_TOL. See gates/g2_gradcheck.py for why the
    eligibility floor exists and why raising REL_TOL is never the fix.

    sim_time_s defaults to 0.15e-12 so this gate stays fast; that is well
    short of the 0.8e-12 production scale, so it cannot by itself catch a
    truncation-error failure that only shows up at production settings.
    Production-scale gradcheck validation runs via scripts/15's own
    `--gradcheck` on the actual recipe (see script 15's `gradcheck()`), not
    here. Set INVDX_G2_SIM_TIME_S to override this default for local
    debugging.
    """
    import os

    import jax.numpy as jnp

    sim_time_s = float(os.environ.get("INVDX_G2_SIM_TIME_S", 0.15e-12))
    pcfg = GratingCouplerConfig(spacing_um=0.020, sim_time_s=sim_time_s,
                                theta_deg=10.0)
    pcfg.design_grid_per_um = 50
    vg_fn, _, _, params, device, value_fn = make_ce_value_and_grad(
        pcfg, p_in=1.0, num_checkpoints=10)

    rho = rasterize_teeth(pcfg, uniform_grating_teeth(
        pcfg, period=0.575, duty=0.5))
    base = (0.1 + 0.8 * rho).reshape(params[device.name].shape)
    beta = jnp.asarray(float(pcfg.beta_schedule[0]), dtype=jnp.float32)

    # The cast belongs to the problem, not the gate: float64-then-cast and
    # float32-throughout do not round identically, and the gate must not be
    # the thing that decides which one this path gets.
    def vg(p, b):
        return vg_fn(jnp.asarray(p, dtype=jnp.float32), b)

    def value(p, b):
        return float(value_fn(jnp.asarray(p, dtype=jnp.float32), b))

    return GradcheckCase(
        vg_fn=vg, value_fn=value, base=base, beta=beta, seed=pcfg.seed,
        info={"spacing_um": pcfg.spacing_um, "sim_time_s": sim_time_s,
              "design_grid_per_um": pcfg.design_grid_per_um})


def reciprocity_case():
    """G4's case: the same uniform grating measured from both sides.

    Cheap settings — 25 nm grid, uniform grating, theta=10 where CE is
    strong:

    forward:  wg-side beam -> forward-TE0-normalized CE into the tilted
              upward Gaussian (wg_side_characterize; injection impurity is
              filtered by the forward mode overlap)
    reverse:  fiber-side tilted beam -> CE into the -x TE0 (characterize)

    The two runs share no normalization: `wg_side_characterize` divides by a
    traced forward TE0 overlap, `characterize` by an empty-cell beam power
    from `calibrated_beam`. That independence is the whole gate — a factor
    applied to both would cancel and stay invisible.
    """
    pcfg = GratingCouplerConfig(spacing_um=0.025, sim_time_s=0.8e-12,
                                theta_deg=10.0)
    teeth = uniform_grating_teeth(pcfg, period=0.575, duty=0.5)
    fwd = wg_side_characterize(pcfg, teeth)
    rev = characterize(pcfg, teeth)
    return ReciprocityCase(
        fwd_dB=fwd["CE_fwd_dB"], rev_dB=rev["CE_dB"],
        extra={"S11_dB": fwd["S11_dB"]})


PROBLEM = ProblemSpec(
    name="grating_coupler",
    config_cls=GratingCouplerConfig,
    gradcheck_case=gradcheck_case,
    reciprocity_case=reciprocity_case,
)
