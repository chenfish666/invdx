"""The 2D free-form path: filter, scene, traced FOM.

Three layers, each with its own acceptance criterion:

  * the 2D conic filter against HAND-COMPUTED kernel values and its
    contractual properties (constant preservation, boundary renormalization,
    non-separability), plus numpy <-> jax parity;
  * the placed Device's transform chain (ConicFilter2D -> TanhProjection)
    against the numpy reference chain, through real fdtdx placement (this is
    what pins _single_voxel_size wiring and the (nx, ny, 1) latent shape);
  * the traced plane-overlap CE (`ce_from_arrays_3d`) against the numpy
    measurement chain (`combine_3d`) on identical synthetic phasor planes —
    tolerance below.

Everything here is CPU-only. The 1D path's tests are untouched elsewhere;
nothing in this file exercises or modifies the 1D behavior.
"""

import numpy as np
import pytest

from invdx.fab import filters_np
from invdx.problems import grating_coupler

# Tolerance for the traced-vs-numpy comparison: the traced chain runs in
# float32/complex64 while the numpy chain runs in float64/complex128, so
# exact equality is impossible; 5e-5 relative is ~50x the observed gap and
# far below any physically meaningful CE difference.
TRACED_VS_NUMPY_RTOL = 5e-5


def tiny_cfg():
    """A grid-legal miniature 3D config (CPU-friendly)."""
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.25, sim_time_s=0.05e-12)
    cfg.design_grid_per_um = 4     # design pixel 0.25 um == one cell
    cfg.L_design = 2.0
    cfg.L_design_y = 1.0
    cfg.pad_x = 1.0
    cfg.dpml = 0.5
    cfg.t_box = 1.0
    cfg.t_sub = 0.5
    cfg.air_above = 2.0
    cfg.min_feature = 0.5          # filter radius 2 pixels -> a real kernel
    cfg.w0 = 1.0
    cfg.x_mon_wg = -1.5
    cfg.fiber_line_y = 1.0
    cfg.src_beam_y = 1.6
    cfg.wg_mon_height = 1.0
    return cfg


# ---------------------------------------------------------------------------
# 2D conic kernel + filter (numpy reference)
# ---------------------------------------------------------------------------


def test_conic_kernel_2d_hand_values():
    # R = 2, dx = dy = 1: taps at integer offsets, weight 1 - sqrt(i^2+j^2)/2
    K = filters_np.conic_kernel_2d(2.0, 1.0, 1.0)
    assert K.shape == (5, 5)
    c = 2  # center index
    assert K[c, c] == pytest.approx(1.0)
    assert K[c + 1, c] == pytest.approx(0.5)          # dist 1
    assert K[c, c + 1] == pytest.approx(0.5)
    assert K[c + 1, c + 1] == pytest.approx(1.0 - np.sqrt(2.0) / 2.0)
    assert K[c + 2, c] == pytest.approx(0.0)          # dist 2 == R exactly
    assert K[c + 2, c + 1] == pytest.approx(0.0)      # dist sqrt(5) > R
    assert K[c + 2, c + 2] == pytest.approx(0.0)      # dist 2*sqrt(2) > R
    # symmetric under 180-degree rotation (what makes corr == conv)
    np.testing.assert_allclose(K, K[::-1, ::-1])


def test_conic_kernel_2d_anisotropic_pixels_isotropic_in_um():
    # dx = 0.5, dy = 1.0, R = 1.0: elliptical in index space, radial in um
    K = filters_np.conic_kernel_2d(1.0, 0.5, 1.0)
    assert K.shape == (5, 3)                          # ri = 2, rj = 1
    ci, cj = 2, 1
    assert K[ci + 2, cj] == pytest.approx(0.0)        # 2 * 0.5 um == R
    assert K[ci + 1, cj] == pytest.approx(0.5)        # 0.5 um
    assert K[ci, cj + 1] == pytest.approx(0.0)        # 1.0 um == R
    # dist sqrt(0.5^2 + 1^2) = 1.118 um > R -> raw weight would be negative,
    # the max(0, .) clip must zero it
    assert K[ci + 1, cj + 1] == pytest.approx(0.0)


def test_conic_filter_2d_preserves_constants():
    # boundary renormalization: filter(c * ones) == c * ones EXACTLY (up to
    # float roundoff), including at edges and corners — same contract as the
    # 1D row-normalized matrix
    x = 0.7 * np.ones((9, 6))
    out = filters_np.conic_filter_2d(x, 0.5, 0.25, 0.25)
    np.testing.assert_allclose(out, x, rtol=0, atol=1e-12)


def test_conic_filter_2d_interior_delta_is_normalized_kernel():
    K = filters_np.conic_kernel_2d(0.5, 0.25, 0.25)   # 5x5 taps
    x = np.zeros((11, 11))
    x[5, 5] = 1.0
    out = filters_np.conic_filter_2d(x, 0.5, 0.25, 0.25)
    # interior pixels: norm == K.sum(), so the delta responds with K/K.sum()
    np.testing.assert_allclose(out[3:8, 3:8], K / K.sum(), atol=1e-12)
    assert out[0, 0] == 0.0


def test_conic_filter_2d_is_not_two_1d_passes():
    # The Euclidean cone is NOT separable: a genuine 2D convolution must
    # differ from an x-pass followed by a y-pass of the 1D cone. Separable
    # passes give a square-ish support whose minimum feature size is
    # direction-dependent, so the radial cone is pinned as a test.
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(12, 12))
    out2d = filters_np.conic_filter_2d(x, 0.5, 0.25, 0.25)

    W = filters_np.conic_filter_matrix(12, 0.5, 4.0)  # 1D, same radius/grid
    sep = W @ x @ W.T                                 # x-pass then y-pass
    assert np.max(np.abs(out2d - sep)) > 1e-3


def test_conic_filter_2d_np_jax_parity():
    import jax.numpy as jnp

    from invdx.fab.filters_jax import make_conic_filter_2d

    rng = np.random.default_rng(1)
    x = rng.uniform(size=(9, 7))
    ref = filters_np.conic_filter_2d(x, 0.5, 0.25, 0.25)
    filt = make_conic_filter_2d((9, 7), 0.5, 0.25, 0.25)
    out = np.asarray(filt(jnp.asarray(x, dtype=jnp.float32)))
    np.testing.assert_allclose(out, ref, rtol=0, atol=5e-6)


# ---------------------------------------------------------------------------
# grid guard (2D twin; the 1D guard is pinned by test_grating_coupler_optimize)
# ---------------------------------------------------------------------------


def test_grid_guard_2d_accepts_clean_grid():
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.020)
    cfg.design_grid_per_um = 50
    grating_coupler.assert_design_grid_snaps_2d(cfg)             # 10.0 / 10.0 / 0.22 ok


def test_grid_guard_2d_rejects_bad_L_design_y():
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.020)
    cfg.design_grid_per_um = 50
    cfg.L_design_y = 1.001
    with pytest.raises(ValueError, match="L_design_y"):
        grating_coupler.assert_design_grid_snaps_2d(cfg)
    # the y leg is NOT relaxable — allow_t_si_snap must not open it
    with pytest.raises(ValueError, match="L_design_y"):
        grating_coupler.assert_design_grid_snaps_2d(cfg, allow_t_si_snap=True)


def test_grid_guard_2d_t_si_leg_relaxable_only_with_flag(capsys):
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.10)            # t_si: 2.2 cells
    cfg.design_grid_per_um = 10
    with pytest.raises(ValueError, match="t_si"):
        grating_coupler.assert_design_grid_snaps_2d(cfg)
    grating_coupler.assert_design_grid_snaps_2d(cfg, allow_t_si_snap=True)
    assert "WARNING" in capsys.readouterr().out


def test_grid_guard_2d_rejects_non_integer_grid():
    cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.044)
    cfg.design_grid_per_um = 250 / 11                 # not a whole number
    with pytest.raises(ValueError, match="positive integer"):
        grating_coupler.assert_design_grid_snaps_2d(cfg)


# ---------------------------------------------------------------------------
# placed Device: transform chain vs the numpy reference chain
# ---------------------------------------------------------------------------


def _place_tiny_design_scene(cfg, with_transforms=True):
    import jax
    import fdtdx

    sim_config, objs, cons, template = grating_coupler.build_scene_design_3d(
        cfg, num_checkpoints=2, wg_width_um=2.0,
        with_transforms=with_transforms, allow_t_si_snap=True)
    key = jax.random.PRNGKey(0)
    key, k1, _ = jax.random.split(key, 3)
    objects, arrays, params, sim_config, _ = fdtdx.place_objects(
        object_list=objs, config=sim_config, constraints=cons, key=k1)
    device = next(d for d in objects.devices if d.name == template.name)
    return sim_config, objects, arrays, params, device


def test_design_shape_2d():
    cfg = tiny_cfg()
    assert grating_coupler.design_shape_2d(cfg) == (8, 4)


def test_placed_device_latent_shape_and_gradient_config():
    cfg = tiny_cfg()
    sim_config, objects, arrays, params, device = _place_tiny_design_scene(cfg)
    nx, ny = grating_coupler.design_shape_2d(cfg)
    assert tuple(params[device.name].shape) == (nx, ny, 1)
    assert sim_config.gradient_config is not None
    assert sim_config.gradient_config.method == "checkpointed"
    # wg_mon must be a PLANE (real y extent), not the quasi-2D thin line
    ph = arrays.detector_states["wg_mon"]["phasor"]
    assert ph.shape[-2] > 4          # ny (quasi-2D would give n_y_cells == 4)


def test_placed_device_chain_matches_numpy_reference():
    import jax.numpy as jnp

    cfg = tiny_cfg()
    _, _, _, _, device = _place_tiny_design_scene(cfg)
    nx, ny = grating_coupler.design_shape_2d(cfg)

    rng = np.random.default_rng(2)
    p = rng.uniform(size=(nx, ny, 1))
    beta = 8.0
    dens = grating_coupler.rho_from_params_2d(
        device, jnp.asarray(p, dtype=jnp.float32), beta)

    pixel = 1.0 / cfg.design_grid_per_um
    xt = filters_np.conic_filter_2d(p[:, :, 0], cfg.filter_radius,
                                    pixel, pixel)
    ref = filters_np.tanh_projection(xt, beta, cfg.eta_i)

    assert dens.shape == (nx, ny)
    np.testing.assert_allclose(dens, ref, rtol=0, atol=2e-5)


# ---------------------------------------------------------------------------
# traced plane-overlap CE vs the numpy measurement chain (combine_3d)
# ---------------------------------------------------------------------------


def test_traced_ce_matches_combine_3d_on_synthetic_planes():
    import types

    import jax.numpy as jnp

    cfg = tiny_cfg()
    wg_width = 2.0
    ny, nz = 14, 10
    rng = np.random.default_rng(3)

    def cplanes(shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape))

    wg_planes = cplanes((1, 1, 2, 1, ny, nz))         # grating-run wg_mon
    fiber_planes = cplanes((1, 1, 2, 12, 12, 1))      # empty-run fiber_mon

    # numpy chain of record
    out = grating_coupler.combine_3d(fiber_planes, wg_planes, cfg, [cfg.lam_c],
                          wg_width_um=wg_width)
    ce_np = out["spectrum"][0]["CE"]
    p_in_np = out["spectrum"][0]["P_in"]

    # static P_in twin (beam_power_3d's arithmetic on the same planes)
    dA = cfg.spacing_um ** 2
    Ey = np.squeeze(fiber_planes[0, 0, 0])
    Hx = np.squeeze(fiber_planes[0, 0, 1])
    p_in = float(abs(0.5 * np.real(np.sum(Ey * np.conj(Hx))) * dA))
    assert p_in == pytest.approx(p_in_np, rel=1e-12)

    # traced twin on the identical wg planes
    target = grating_coupler.wg_mode_target_on_monitor_3d(cfg, ny, nz,
                                               wg_width_um=wg_width)
    fake = types.SimpleNamespace(detector_states={
        "wg_mon": {"phasor": jnp.asarray(wg_planes)}})
    ce_j = float(grating_coupler.ce_from_arrays_3d(fake, cfg, target, p_in))

    assert ce_j == pytest.approx(ce_np, rel=TRACED_VS_NUMPY_RTOL)


def test_wg_mode_target_matches_wg_mode_3d():
    # the target builder must sample the mode EXACTLY as combine_3d does
    cfg = tiny_cfg()
    ny, nz = 14, 10
    Em_t, Hm_t, Pm, neff_t = grating_coupler.wg_mode_target_on_monitor_3d(
        cfg, ny, nz, wg_width_um=2.0)

    z_mon_lo = cfg.t_si / 2 - cfg.wg_mon_height / 2
    zs = z_mon_lo + (np.arange(nz) + 0.5) * cfg.spacing_um
    ys = (np.arange(ny) - ny / 2 + 0.5) * cfg.spacing_um
    Em, Hm_fwd, neff = grating_coupler.wg_mode_3d(ys, zs, cfg, 2.0)

    np.testing.assert_allclose(Em_t, Em)
    np.testing.assert_allclose(Hm_t, -Hm_fwd)
    assert neff_t == pytest.approx(neff)
    dA = cfg.spacing_um ** 2
    assert Pm == pytest.approx(
        abs(0.5 * np.real(np.sum(Em * np.conj(-Hm_fwd))) * dA))


def test_make_ce_value_and_grad_3d_rejects_w_s11():
    cfg = tiny_cfg()
    cfg.w_s11 = 0.3
    with pytest.raises(NotImplementedError, match="w_s11"):
        grating_coupler.make_ce_value_and_grad_3d(cfg, 1.0)
