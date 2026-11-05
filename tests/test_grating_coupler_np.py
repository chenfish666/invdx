"""grating_coupler_np -- the numpy-pure half of grating_coupler -- and the
Meep reference built on it (engines/meep_worker.py task `grating_coupler`,
scripts/03).

No Meep is needed here. What can be pinned without it is pinned: that the
module and the worker import inside a Meep-like environment, that the device
and the planes the Meep job is handed are the ones the fdtdx scene places,
that the beam tilts the way fdtdx's does, that both engines' CE goes through
one normalization with the right H sign for each, and the ridge arithmetic
the acceptance is defined on.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types

import numpy as np
import pytest

from invdx.engines import meep_worker
from invdx.problems import grating_coupler_np as gc

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
TEETH_KW = dict(period=0.575, duty=0.5)


def _cfg(**kw):
    kw.setdefault("theta_deg", 10.0)
    return gc.GratingCouplerConfig(**kw)


# --------------------------------------------------------------------------
# import discipline: what runs inside the Meep environment
# --------------------------------------------------------------------------


def test_meep_side_imports_without_jax_fdtdx_autograd_or_meep():
    """The Meep environment has no jax, fdtdx or autograd, and the parent
    side builds the payload without meep. Blocking all four and importing
    what the worker imports is the closest this suite gets to that
    environment -- a stray top-level import in grating_coupler_np, or in the
    worker, fails here instead of in a Meep job."""
    code = textwrap.dedent("""
        import importlib.abc, json, sys
        BLOCKED = {"jax", "jaxlib", "fdtdx", "autograd", "optax", "meep"}

        class NotInMeepEnv(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(name + " is not importable here")
                return None

        sys.meta_path.insert(0, NotInMeepEnv())
        from invdx.problems import grating_coupler_np as gc
        from invdx.engines import conventions, meep_worker
        assert "grating_coupler" in meep_worker.TASKS
        cfg = gc.GratingCouplerConfig(theta_deg=10.0)
        teeth = gc.uniform_grating_teeth(cfg, period=0.575, duty=0.5)
        payload = gc.meep_payload(cfg, teeth, [1.30, 1.31, 1.32])
        json.dumps(payload)
        meep_worker._beam_points(gc, payload["beam"])
        loaded = sorted(m for m in sys.modules
                        if m.split(".")[0] in BLOCKED
                        or m == "invdx.problems.grating_coupler")
        assert not loaded, loaded
        print("clean")
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().endswith("clean")


def test_grating_coupler_reexports_every_public_np_name():
    """`grating_coupler.X` keeps working for everything that moved, and for
    everything added here later: the re-export list is checked against the
    module, not against a second hand-kept list."""
    pytest.importorskip("fdtdx")
    from invdx.problems import grating_coupler

    public = [n for n, obj in vars(gc).items() if not n.startswith("_")
              and (getattr(obj, "__module__", None) == gc.__name__
                   or n.isupper())]
    assert "GratingCouplerConfig" in public and "ce_sample" in public
    missing = [n for n in public
               if getattr(grating_coupler, n, None) is not getattr(gc, n)]
    assert not missing, f"not re-exported by grating_coupler: {missing}"


# --------------------------------------------------------------------------
# the device and its planes: one definition, checked against fdtdx's scene
# --------------------------------------------------------------------------


def test_chip_blocks_default_device():
    cfg = _cfg()
    teeth = gc.uniform_grating_teeth(cfg, **TEETH_KW)
    blocks = {b.name: b for b in gc.chip_blocks(cfg, teeth)}
    assert list(blocks)[:3] == ["substrate", "box", "wg_slab"]
    sub, box, slab = blocks["substrate"], blocks["box"], blocks["wg_slab"]
    assert (sub.material, sub.y_min, sub.y_min + sub.y_size) == ("si", -5.5, -3.0)
    assert (box.material, box.y_min, box.y_min + box.y_size) == ("sio2", -3.0, 0.0)
    assert (slab.x_min, slab.x_min + slab.x_size) == (-10.0, -5.0)
    tooth = [b for n, b in blocks.items() if n.startswith("tooth")]
    assert len(tooth) == 17
    assert tooth[0].x_min == pytest.approx(-5.0)
    assert all(b.x_size == pytest.approx(0.2875) and b.y_size == 0.22
               for b in tooth)
    assert gc.material_index(cfg, "si") == cfg.n_si
    assert gc.material_index(cfg, "sio2") == cfg.n_sio2


def test_fdtdx_scene_places_the_np_device_and_planes():
    """Place the real fdtdx scene and read back where every block and plane
    landed. The grid (50 nm) and the grating (0.6 um, duty 0.5) are chosen
    so no edge falls on a half cell, where rounding could go either way."""
    pytest.importorskip("fdtdx")
    jax = pytest.importorskip("jax")
    import fdtdx

    from invdx.problems import grating_coupler

    cfg = _cfg(spacing_um=0.05)
    teeth = gc.uniform_grating_teeth(cfg, period=0.6, duty=0.5)
    sim_config, objs, cons = grating_coupler.build_scene(cfg, teeth=teeth)
    objects, *_ = fdtdx.place_objects(object_list=objs, config=sim_config,
                                      constraints=cons,
                                      key=jax.random.PRNGKey(0))
    placed = {o.name: o.grid_slice_tuple for o in objects.object_list}
    dx = cfg.spacing_um

    def cells(lo_um, size_um):
        # a plane (zero extent in ports) occupies one fdtdx cell
        n = max(1, round(size_um / dx))
        return round(lo_um / dx), round(lo_um / dx) + n

    blocks = gc.chip_blocks(cfg, teeth)
    assert {b.name for b in blocks} <= set(placed)
    for b in blocks:
        sx, _, sz = placed[b.name]
        assert tuple(sx) == cells(cfg.X0 + b.x_min, b.x_size), b.name
        assert tuple(sz) == cells(cfg.Z0 + b.y_min, b.y_size), b.name
    for name, ln in gc.ports(cfg).items():
        sx, _, sz = placed[name]
        assert tuple(sx) == cells(cfg.X0 + ln.x - ln.size_x / 2,
                                  ln.size_x), name
        assert tuple(sz) == cells(cfg.Z0 + ln.y - ln.size_y / 2,
                                  ln.size_y), name


def test_beam_tilt_and_delay_match_fdtdx():
    """fiber_beam_profile's claims about the pinned fdtdx, checked against
    fdtdx's own functions: azimuth = -theta on a z-plane source with
    direction "-" gives the wave vector (+sin, 0, -cos), and each cell
    launches (x - x0) sin(theta)/c late -- the delay the Meep beam copies."""
    pytest.importorskip("fdtdx")
    jnp = pytest.importorskip("jax.numpy")
    from fdtdx.core.grid import calculate_time_offset_yee
    from fdtdx.core.linalg import rotate_vector

    theta = 10.0
    k = np.asarray(rotate_vector(jnp.array([0.0, 0.0, -1.0]),
                                 np.deg2rad(-theta), 0.0, (0, 1, 2)))
    th = np.deg2rad(theta)
    assert k == pytest.approx([np.sin(th), 0.0, -np.cos(th)], abs=1e-6)

    nx, dx, dt, c = 9, 1e-8, 1e-17, 299792458.0
    off_E, _ = calculate_time_offset_yee(
        center=jnp.array([4.0, 0.5]), wave_vector=jnp.asarray(k),
        inv_permittivities=jnp.ones((1, nx, 2, 1)), inv_permeabilities=1.0,
        resolution=dx, time_step_duration=dt)
    # fdtdx evaluates the profile at (step + offset) * dt, so a launch delay
    # of tau is an offset of -tau / dt
    fdtdx_delay_s = -np.asarray(off_E[1, :, 0, 0]) * dt
    xs_um = (np.arange(nx) - 4.0) * dx * 1e6
    _, delay_um = gc.fiber_beam_profile(xs_um, 0.0, 4.6, theta)
    assert delay_um * 1e-6 / c == pytest.approx(fdtdx_delay_s, rel=1e-4)
    assert np.all(np.diff(delay_um) > 0)          # +x launches later


# --------------------------------------------------------------------------
# one normalization for both engines
# --------------------------------------------------------------------------


def _line_fields(lam=1.31, n=400, seed=0):
    rng = np.random.default_rng(seed)
    cfg = _cfg()
    xs = np.linspace(-8.75, 8.75, 1401)
    E_fib = np.exp(-(xs / 4.6) ** 2) * np.exp(-0.3j * xs)
    H_fib = 0.98 * E_fib
    zs = cfg.t_si / 2 - cfg.wg_mon_height / 2 + (np.arange(n) + 0.5) * 0.00625
    E_wg = rng.normal(size=n) + 1j * rng.normal(size=n)
    H_wg = rng.normal(size=n) + 1j * rng.normal(size=n)
    return cfg, E_fib, H_fib, E_wg, H_wg, zs


def test_ce_sample_is_the_old_characterize_spectrum_arithmetic():
    """Bitwise: ce_sample replaced the loop body of characterize_spectrum,
    and earlier fdtdx numbers must not move by an ulp because of that."""
    cfg, E_fib, H_fib, E_wg, H_wg, zs = _line_fields()
    dl, lam = 0.00625, 1.297
    row = gc.ce_sample(lam, E_fib, H_fib, E_wg, H_wg, zs, dl, t_si=cfg.t_si,
                       n_si=cfg.n_si, n_sio2=cfg.n_sio2)

    # the pre-split loop body, verbatim
    p_in = gc.phasor_line_power(E_fib, H_fib, dl)
    lam_cfg = type(cfg)(**{})
    lam_cfg.lam_c, lam_cfg.t_si = lam, cfg.t_si
    lam_cfg.n_si, lam_cfg.n_sio2 = cfg.n_si, cfg.n_sio2
    Em, Hm_fwd, neff = gc.slab_te0_mode(zs, 0.0, lam_cfg)
    p_mode = gc.overlap_power_directional(E_wg, H_wg, Em, -Hm_fwd, dl)
    ce = p_mode / p_in
    assert row["CE"] == float(ce) and row["P_in"] == p_in
    assert row["P_mode"] == p_mode and row["n_eff"] == float(neff)
    assert row["CE_dB"] == float(10 * np.log10(abs(ce) + 1e-15))


def _meep_slab_mode_fields(cfg, lam, zs, direction, amp=0.7 - 0.2j):
    """The TE0 mode as Meep's DFT would hold it on an x-plane, with Hy taken
    from Maxwell rather than from the sign rule under test: e^{-i w t} (the
    phasor Meep's DFT extracts), Ez ~ e^{i beta x}, and
    Faraday's law  -i w mu0 H = -curl E  ->  Hy = (i / w) dEz/dx  (c = 1),
    differentiated numerically across the plane."""
    neff = gc.slab_te0_neff(lam, cfg.t_si, cfg.n_si, cfg.n_sio2, 1.0)
    w = 2 * np.pi / lam
    beta = direction * w * neff
    mode = types.SimpleNamespace(lam_c=lam, t_si=cfg.t_si, n_si=cfg.n_si,
                                 n_sio2=cfg.n_sio2)
    prof, _, _ = gc.slab_te0_mode(zs, 0.0, mode)
    h = 1e-4

    def ez(x):
        return amp * prof * np.exp(1j * beta * x)

    hy = (1j / w) * (ez(h) - ez(-h)) / (2 * h)
    return ez(0.0), hy


def test_meep_wg_line_sign_accepts_minus_x_and_rejects_plus_x():
    """The worker hands ce_sample -Hy as the wg-line H. With fields that
    obey Meep's Maxwell equations, that choice must give the full power of
    a -x traveling mode and none of a +x one; the opposite sign would
    report the reflection instead of the coupled light."""
    cfg, E_fib, H_fib, _, _, zs = _line_fields(n=800)
    lam, dl = 1.31, 0.00625
    kw = dict(t_si=cfg.t_si, n_si=cfg.n_si, n_sio2=cfg.n_sio2)
    ez, hy = _meep_slab_mode_fields(cfg, lam, zs, direction=-1)
    sx = 0.5 * np.real(np.sum(ez * np.conj(-hy))) * dl
    assert sx < 0                                  # it does travel -x
    back = gc.ce_sample(lam, E_fib, H_fib, ez, -hy, zs, dl, **kw)
    assert back["P_mode"] == pytest.approx(abs(sx), rel=1e-6)
    ez, hy = _meep_slab_mode_fields(cfg, lam, zs, direction=+1)
    fwd = gc.ce_sample(lam, E_fib, H_fib, ez, -hy, zs, dl, **kw)
    assert fwd["P_mode"] < 1e-9 * back["P_mode"]


def test_characterize_spectrum_normalizes_with_ce_sample(monkeypatch):
    """The fdtdx chain's wiring: fed fake phasor arrays, characterize_spectrum
    must return exactly what ce_sample computes from the same lines."""
    pytest.importorskip("fdtdx")
    from invdx.problems import grating_coupler

    cfg = _cfg(spacing_um=0.025)
    lams = [1.30, 1.31, 1.32]
    rng = np.random.default_rng(3)
    nx, ny, nz = 690, 4, 100

    def phasors(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    states = {
        False: {"fiber_mon": {"phasor": phasors((1, 3, 2, nx, ny, 1))}},
        True: {"wg_mon": {"phasor": phasors((1, 3, 2, 1, ny, nz))}},
    }

    def fake_run(cfg_, teeth, with_chip, **kw):
        return types.SimpleNamespace(detector_states=states[with_chip])

    monkeypatch.setattr(grating_coupler, "_run", fake_run)
    out = grating_coupler.characterize_spectrum(cfg, [], lams,
                                                azimuth_sign=1.0)
    fib = states[False]["fiber_mon"]["phasor"]
    wg = states[True]["wg_mon"]["phasor"]
    z_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    zs = z_lo + (np.arange(nz) + 0.5) * cfg.spacing_um
    for k, lam in enumerate(lams):
        want = gc.ce_sample(
            lam, fib[0, k, 0, :, :, 0].mean(axis=1),
            fib[0, k, 1, :, :, 0].mean(axis=1),
            wg[0, k, 0, 0].mean(axis=0), wg[0, k, 1, 0].mean(axis=0), zs,
            cfg.spacing_um, t_si=cfg.t_si, n_si=cfg.n_si, n_sio2=cfg.n_sio2)
        assert out["spectrum"][k] == want



def _worker_fields(walk_sign=+1.0, flux_factor=2.0, lams=(1.30, 1.31, 1.32)):
    """What task_grating_coupler's two Meep runs would record, built from
    Maxwell's equations: a theta-tilted Gaussian on fiber_mon (empty run) and
    a -x traveling TE0 mode on wg_mon (grating run). flux_factor is Meep's
    flux over the physical power (2: lesson 1)."""
    cfg = _cfg()
    payload = gc.meep_payload(cfg, gc.uniform_grating_teeth(cfg, **TEETH_KW),
                              list(lams))
    beam, mons = payload["beam"], payload["monitors"]
    dl = 1.0 / payload["resolution"]
    th = np.deg2rad(beam["theta_deg"])
    walk = (beam["y"] - mons["fiber_mon"]["center"][1]) * np.tan(th)
    xs = np.arange(-8.75, 8.75 + dl / 2, dl)
    ys = np.arange(-1.14, 1.36 + dl / 2, dl)
    empty = {"Ez": [], "Hx": []}
    grating = {"Ez": [], "Hy": []}
    flux, expect = [], []
    for lam in lams:
        k0 = 2 * np.pi / lam
        ez = (np.exp(-(((xs - walk_sign * walk) * np.cos(th) / 4.6) ** 2))
              * np.exp(1j * k0 * np.sin(th) * xs))
        hx = -np.cos(th) * ez
        p_in = abs(0.5 * np.real(np.sum(ez * np.conj(hx))) * dl)
        empty["Ez"].append(ez)
        empty["Hx"].append(hx)
        flux.append(-flux_factor * p_in)               # downward: negative
        ez_w, hy_w = _meep_slab_mode_fields(cfg, lam, ys, direction=-1)
        grating["Ez"].append(ez_w)
        grating["Hy"].append(hy_w)
        sx = 0.5 * np.real(np.sum(ez_w * np.conj(-hy_w))) * dl
        expect.append(abs(sx) / p_in)
    empty = {k: np.array(v) for k, v in empty.items()}
    grating = {k: np.array(v) for k, v in grating.items()}
    return payload, xs, empty, np.array(flux), ys, grating, expect


def test_worker_measures_ce_of_the_minus_x_mode():
    payload, xs, empty, flux, ys, grating, expect = _worker_fields()
    rows, checks = meep_worker._gc_measure(payload, xs, empty, flux, ys,
                                           grating)
    assert [r["CE"] for r in rows] == pytest.approx(expect, rel=1e-6)
    assert checks["flux_over_2P_in_max_dev"] < 1e-12
    assert all(abs(r["beam_centroid_um"] - checks["beam_centroid_expected_um"])
               < 1e-3 for r in rows)


def test_worker_refuses_a_beam_tilted_toward_the_waveguide():
    payload, xs, empty, flux, ys, grating, _ = _worker_fields(walk_sign=-1.0)
    with pytest.raises(RuntimeError, match="toward \\+x"):
        meep_worker._gc_measure(payload, xs, empty, flux, ys, grating)


def test_worker_refuses_a_flux_that_breaks_lesson_1():
    payload, xs, empty, flux, ys, grating, _ = _worker_fields(flux_factor=1.0)
    with pytest.raises(RuntimeError, match="lesson 1"):
        meep_worker._gc_measure(payload, xs, empty, flux, ys, grating)


# --------------------------------------------------------------------------
# the ridge
# --------------------------------------------------------------------------


def _gaussian_ridge(lams, lam0, peak_db, fwhm_um=0.05):
    sig = fwhm_um / (2 * np.sqrt(2 * np.log(2)))
    ce = 10 ** (peak_db / 10) * np.exp(-((np.asarray(lams) - lam0) ** 2)
                                       / (2 * sig ** 2))
    return [{"lam_um": float(l), "CE_dB": float(10 * np.log10(c))}
            for l, c in zip(lams, ce)]


def test_ridge_recovers_a_gaussian_ridge_between_samples():
    lams = np.linspace(1.26, 1.36, 21)                 # 5 nm samples
    spec = _gaussian_ridge(lams, 1.3123, -9.7)
    r = gc.ridge(spec[::-1])                            # order is irrelevant
    assert r["resolved"] and r["note"] is None
    assert r["lam_um"] == pytest.approx(1.3123, abs=1e-9)
    assert r["CE_dB"] == pytest.approx(-9.7, abs=1e-9)
    assert r["sample_lam_um"] == pytest.approx(1.31)
    assert r["sample_CE_dB"] < r["CE_dB"]


def test_ridge_on_the_band_edge_is_a_bound_not_a_ridge():
    lams = np.linspace(1.26, 1.36, 11)
    r = gc.ridge(_gaussian_ridge(lams, 1.40, -8.0))
    assert not r["resolved"]
    assert r["lam_um"] == pytest.approx(1.36) and "long-wavelength" in r["note"]
    with pytest.raises(ValueError):
        gc.ridge(_gaussian_ridge([1.30, 1.31], 1.30, -8.0))


def test_compare_ridges_gates_the_peak_and_reports_the_wavelength():
    lams = np.linspace(1.26, 1.36, 21)
    ref = _gaussian_ridge(lams, 1.310, -10.0)
    near = gc.compare_ridges(ref, _gaussian_ridge(lams, 1.330, -11.5), 2.0)
    assert near["pass"] and near["reason"] is None
    assert near["delta_peak_dB"] == pytest.approx(-1.5, abs=1e-9)
    assert near["delta_lam_nm"] == pytest.approx(20.0, abs=1e-6)  # not gated
    far = gc.compare_ridges(ref, _gaussian_ridge(lams, 1.310, -12.5), 2.0)
    assert not far["pass"] and "-2.50 dB" in far["reason"]
    edge = gc.compare_ridges(ref, _gaussian_ridge(lams, 1.45, -10.0), 2.0)
    assert not edge["pass"] and "candidate" in edge["reason"]


# --------------------------------------------------------------------------
# the Meep job description
# --------------------------------------------------------------------------


def _payload(**kw):
    cfg = _cfg(**kw)
    return gc.meep_payload(cfg, gc.uniform_grating_teeth(cfg, **TEETH_KW),
                           list(np.linspace(1.26, 1.36, 21)))


def test_payload_is_plain_json_and_sits_on_the_fdtdx_grid():
    cfg = _cfg()
    p = _payload()
    assert json.loads(json.dumps(p)) == p
    res = p["resolution"]
    assert res == cfg.resolution == round(1 / cfg.spacing_um)
    sx, sy = p["cell"]["size"]
    cx, cy = p["cell"]["center"]
    # fdtdx's pixel counts and fdtdx's lower-left corner (-X0, -Z0)
    assert (round(sx * res), round(sy * res)) == (
        round(cfg.cell_x / cfg.spacing_um), round(cfg.cell_z / cfg.spacing_um))
    assert (cx - sx / 2, cy - sy / 2) == pytest.approx((-cfg.X0, -cfg.Z0))
    # every tooth edge and the slab bottom on a Meep grid line
    for b in p["blocks"]:
        if b["name"].startswith("tooth"):
            for e in (b["center"][0] - b["size"][0] / 2,
                      b["center"][0] + b["size"][0] / 2):
                assert (e + cfg.X0) * res == pytest.approx(
                    round((e + cfg.X0) * res), abs=1e-6)


def test_payload_carries_edge_blocks_through_the_cell_edge():
    cfg = _cfg()
    p = _payload()
    blocks = {b["name"]: b for b in p["blocks"]}
    sx, sy = p["cell"]["size"]
    cx, cy = p["cell"]["center"]

    def span(b, ax):
        return (b["center"][ax] - b["size"][ax] / 2,
                b["center"][ax] + b["size"][ax] / 2)

    for name in ("substrate", "box"):
        lo, hi = span(blocks[name], 0)
        assert lo < cx - sx / 2 and hi > cx + sx / 2, name
    assert span(blocks["substrate"], 1)[0] < cy - sy / 2
    assert span(blocks["wg_slab"], 0) == pytest.approx(
        (-cfg.X0 - cfg.dpml, -cfg.L_design / 2))
    assert span(blocks["box"], 1) == pytest.approx((-cfg.t_box, 0.0))
    assert span(blocks["tooth0"], 0) == pytest.approx((-5.0, -4.7125))
    assert p["monitors"]["fiber_mon"]["center"] == pytest.approx([0.0, 0.8])
    assert p["monitors"]["wg_mon"]["size"] == [0.0, cfg.wg_mon_height]


def test_payload_ignores_fdtdx_only_fields_and_follows_the_device():
    """A stored reference stays valid across fdtdx-only changes (the
    scripts/03 --meep-json rule) and stops being valid when the device,
    the beam, the grid or the wavelengths change."""
    base = _payload()
    for kw in (dict(spacing_um=0.01), dict(n_y_cells=8),
               dict(sim_time_s=0.8e-12), dict(seed=7)):
        assert _payload(**kw) == base, kw
    for kw in (dict(t_si=0.225), dict(theta_deg=8.0), dict(resolution=40),
               dict(w0=5.2), dict(dft_decay_tol=1e-8)):
        assert _payload(**kw) != base, kw


def test_beam_points_launch_the_tilted_waist_one_pixel_apart():
    p = _payload()
    beam = p["beam"]
    xs, amp, t0 = meep_worker._beam_points(gc, beam)
    assert xs.size == round(beam["width"] / beam["pitch"]) == 1360
    assert np.diff(xs) == pytest.approx(beam["pitch"])
    assert xs.mean() == pytest.approx(beam["x0"], abs=1e-12)
    th = np.deg2rad(beam["theta_deg"])
    assert amp == pytest.approx(
        np.exp(-(((xs - beam["x0"]) * np.cos(th) / beam["w0"]) ** 2)))
    assert t0.min() == 0.0 and np.all(np.diff(t0) > 0)     # +x launches later
    assert t0 == pytest.approx((xs - xs[0]) * np.sin(th))


# --------------------------------------------------------------------------
# scripts/03: the comparison, with both engines faked
# --------------------------------------------------------------------------

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "scripts", "03_grating_coupler_baseline.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("_baseline_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_meep(lam0=1.310, peak_db=-10.0):
    def meep_spectrum(cfg, teeth, lams, n_ranks=2, timeout=0):
        return {"engine": "meep", "meep_version": "fake", "processes": n_ranks,
                "resolution": cfg.resolution,
                "spectrum": _gaussian_ridge(lams, lam0, peak_db),
                "checks": {"flux_over_2P_in_max_dev": 0.0},
                "payload": gc.meep_payload(cfg, teeth, lams)}
    return meep_spectrum


def _run_script(mod, tmp_path, *argv):
    old = sys.argv
    sys.argv = ["03_grating_coupler_baseline.py", "--set",
                f"runs_root={tmp_path}", *argv]
    try:
        mod.main()
    finally:
        sys.argv = old
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    return run_dir


def test_meep_only_regenerates_the_reference_without_jax(tmp_path):
    code = textwrap.dedent(f"""
        import importlib.abc, importlib.util, sys
        BLOCKED = {{"jax", "jaxlib", "fdtdx", "autograd", "optax", "meep"}}

        class NotHere(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(name)
                return None

        sys.meta_path.insert(0, NotHere())
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        import test_grating_coupler_np as t
        mod = t._load_script()
        mod.gc.meep_spectrum = t._fake_meep()
        sys.argv = ["03", "--meep-only", "--set", "runs_root={tmp_path}"]
        mod.main()
        assert "invdx.problems.grating_coupler" not in sys.modules
        print("clean")
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    meep = json.loads((run_dir / "meep.json").read_text())
    assert meep["payload"]["beam"]["theta_deg"] == 10.0
    assert not (run_dir / "results.json").exists()


def _fake_fdtdx(monkeypatch, lam0, peak_db):
    pytest.importorskip("fdtdx")
    from invdx.problems import grating_coupler

    monkeypatch.setattr(
        grating_coupler, "characterize_spectrum",
        lambda cfg, teeth, lams, **kw: {"spectrum": _gaussian_ridge(
            lams, lam0, peak_db), "azimuth_sign": 1.0})
    monkeypatch.setattr(grating_coupler, "characterize",
                        lambda cfg, teeth, **kw: {"CE_dB": -31.0})


def test_script_passes_on_agreeing_ridges(monkeypatch, tmp_path):
    mod = _load_script()
    monkeypatch.setattr(mod.gc, "meep_spectrum", _fake_meep(1.310, -10.0))
    _fake_fdtdx(monkeypatch, 1.322, -10.9)
    run_dir = _run_script(mod, tmp_path)
    res = json.loads((run_dir / "results.json").read_text())
    cmp = res["theta_10_ridge"]
    assert cmp["pass"]
    assert cmp["delta_peak_dB"] == pytest.approx(-0.9, abs=1e-6)
    assert cmp["delta_lam_nm"] == pytest.approx(12.0, abs=1e-3)
    assert res["historical_reference"]["CE_dB"] == -10.1
    assert res["effective_geometry"]["fdtdx_t_si_um"] == pytest.approx(0.225)


def test_script_fails_on_a_peak_gap_and_exits_1(monkeypatch, tmp_path):
    mod = _load_script()
    monkeypatch.setattr(mod.gc, "meep_spectrum", _fake_meep(1.310, -10.0))
    _fake_fdtdx(monkeypatch, 1.310, -12.6)
    with pytest.raises(SystemExit) as exc:
        _run_script(mod, tmp_path)
    assert exc.value.code == 1


def test_meep_json_is_refused_for_another_device(monkeypatch, tmp_path):
    mod = _load_script()
    # the payload the script sends with its own defaults
    cfg = _cfg(dft_decay_tol=mod.MEEP_DECAY_TOL)
    lams = mod.parse_lams(mod.DEFAULT_LAMS)
    stored = _fake_meep()(cfg, gc.uniform_grating_teeth(cfg, **TEETH_KW),
                          lams)
    ref = tmp_path / "ref" / "meep.json"
    ref.parent.mkdir()
    ref.write_text(json.dumps(stored))
    runs = tmp_path / "runs"
    runs.mkdir()
    _fake_fdtdx(monkeypatch, 1.312, -10.2)
    monkeypatch.setattr(mod.gc, "meep_spectrum",
                        lambda *a, **k: pytest.fail("must reuse, not run"))
    run_dir = _run_script(mod, runs, "--meep-json", str(ref),
                          "--set", "spacing_um=0.01")   # fdtdx-only change
    assert json.loads((run_dir / "results.json").read_text())[
        "theta_10_ridge"]["pass"]
    with pytest.raises(SystemExit, match="beam.w0"):
        _run_script(mod, tmp_path / "ref", "--meep-json", str(ref),
                    "--set", "w0=5.2")
