"""fdtdx scene factories — the Layer A adapter for the GPU engine.

Canonical flow (from fdtdx's ceviche example): build object_list + placement
constraints -> place_objects -> apply_params -> run_fdtd -> read detector
states. This module provides small, config-driven building blocks; concrete
design problems compose their own scenes on top of these.

All lengths here are um at the interface (config convention) and converted
to meters at the fdtdx boundary.

NOTE: this adapter targets the RELEASED fdtdx==0.6.2 (PyPI), whose
SimulationConfig takes `resolution=<spacing in meters>`. The vendored fdtdx
checkout is a dev snapshot ahead of that release (grid=UniformGrid(...) API,
symmetry, dispersion) — its examples/docs don't fully match the pinned
version. Re-check this adapter when re-pinning to the next release.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

UM = 1e-6


def make_sim_config(cfg, time_s=100e-15):
    dtype = jnp.float32 if cfg.dtype == "float32" else jnp.float64
    return fdtdx.SimulationConfig(
        time=time_s,
        resolution=cfg.spacing_um * UM,
        dtype=dtype,
        courant_factor=0.99,
    )


def vacuum_flux_scene(cfg, wavelength_um=1.55, cell_um=(1.5, 1.5, 1.5),
                      time_s=100e-15, periodic_sides=False, slab=None,
                      with_input_detector=False):
    """Tiny cell: +z plane-wave source and a flux detector ("flux") above it.

    periodic_sides      — periodic x/y boundaries (clean 1D plane wave, the
                          same trick as the meep worker's k_point=0)
    slab                — optional (n, t_um): a lossless dielectric slab at
                          the cell center, spanning the full cross-section
    with_input_detector — adds a second detector "flux_in" just above the
                          source for conservation/normalization checks

    Returns (config, object_list, constraints, detector) — the caller runs
    place_objects/run_fdtd, so gates can reuse the same scene under
    value_and_grad later.
    """
    config = make_sim_config(cfg, time_s=time_s)

    object_list, constraints = [], []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=tuple(s * UM for s in cell_um))
    object_list.append(volume)

    overrides = {}
    if periodic_sides:
        overrides = {k: "periodic" for k in
                     ("min_x", "max_x", "min_y", "max_y")}
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=8, override_types=overrides)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(bound_dict.values())

    if slab is not None:
        n_slab, t_um = slab
        slab_obj = fdtdx.UniformMaterialObject(
            partial_real_shape=(None, None, t_um * UM),
            material=fdtdx.Material(permittivity=n_slab ** 2),
        )
        constraints.append(slab_obj.place_at_center(volume, axes=(0, 1, 2)))
        constraints.append(slab_obj.same_size(volume, axes=(0, 1)))
        object_list.append(slab_obj)

    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength_um * UM),
        direction="+",
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
    )
    constraints.append(source.place_relative_to(
        volume, axes=(2,), own_positions=(-1,), other_positions=(-1,),
        grid_margins=(bound_cfg.thickness_grid_minz + 3,)))
    object_list.append(source)

    detector = fdtdx.PoyntingFluxDetector(
        name="flux",
        partial_grid_shape=(None, None, 1),
        direction="+",
    )
    constraints.append(detector.place_relative_to(
        volume, axes=(2,), own_positions=(1,), other_positions=(1,),
        grid_margins=(-bound_cfg.thickness_grid_maxz - 3,)))
    object_list.append(detector)

    if with_input_detector:
        det_in = fdtdx.PoyntingFluxDetector(
            name="flux_in",
            partial_grid_shape=(None, None, 1),
            direction="+",
        )
        constraints.append(det_in.place_relative_to(
            volume, axes=(2,), own_positions=(-1,), other_positions=(-1,),
            grid_margins=(bound_cfg.thickness_grid_minz + 6,)))
        object_list.append(det_in)

    return config, object_list, constraints, detector


def device_slab_scene(cfg, wavelength_um=1.55, cell_um=(1.0, 1.0, 1.5),
                      time_s=50e-15, n_voxels_x=10):
    """Gradcheck scene: the vacuum cell with an optimizable 1-voxel-thick
    Device slab between source and detector, parameterized along x only
    (n_voxels_x continuous air<->silicon voxels, no transforms).

    Returns (config, object_list, constraints, detector, device); config has
    a checkpointed GradientConfig so run_fdtd is differentiable.
    """
    config, object_list, constraints, detector = vacuum_flux_scene(
        cfg, wavelength_um=wavelength_um, cell_um=cell_um, time_s=time_s)
    config = config.aset("gradient_config", fdtdx.GradientConfig(
        method="checkpointed", num_checkpoints=10))

    slab_t = 2 * cfg.spacing_um  # a couple of cells thick
    device = fdtdx.Device(
        name="slab",
        partial_real_shape=(cell_um[0] * UM, cell_um[1] * UM, slab_t * UM),
        materials={
            "air": fdtdx.Material(permittivity=1.0),
            "si": fdtdx.Material(permittivity=12.25),
        },
        param_transforms=[],
        partial_voxel_real_shape=(cell_um[0] / n_voxels_x * UM,
                                  cell_um[1] * UM, slab_t * UM),
    )
    volume = object_list[0]
    constraints.append(device.place_at_center(volume, axes=(0, 1, 2)))
    object_list.append(device)
    return config, object_list, constraints, detector, device


def run_forward(config, object_list, constraints, seed=0):
    """place_objects -> apply_params -> run_fdtd; returns final arrays."""
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints, key=k1)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, k2)
    _, arrays = fdtdx.run_fdtd(
        arrays=arrays, objects=objects, config=config, key=key)
    return arrays


def steady_flux(arrays, detector, n_avg=50):
    """Mean directional Poynting flux over the last n_avg recorded steps.

    detector may be the detector object or its name string. For CW sources
    pick n_avg close to an integer number of optical periods.
    """
    name = detector if isinstance(detector, str) else detector.name
    flux = np.asarray(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.mean(flux[-n_avg:]))


def gpu_platform():
    """'gpu' when JAX sees a CUDA device, else the default platform name."""
    return jax.devices()[0].platform
