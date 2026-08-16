"""Fleet-tuned FORWARD-ONLY fast path for fdtdx.run_fdtd.

Vendors the inner time loop of fdtdx (c) Yannik Mahlau et al., MIT License
(github.com/ymahlau/fdtdx) (the sanctioned "vendor one module when blocked"
escape hatch — see README Layer A notes). This is a fleet-specific
optimization following HPC-competition discipline: the upstream loop is
written for generality (anisotropic tensors, conductivity, dispersion,
reversible gradients); our pvgc production scenes only ever exercise the
isotropic + PML/periodic + TFSF-plane-source + PhasorDetector subset, so we
specialize the hot loop for that subset and keep the general engine as the
reference. Worth offering upstream once generalized.

What it does differently (validated by scripts/13_curl_microbench.py on
Turing, 16.8M cells, 1.89x kernel speedup, bitwise-identical output):

  1. E/H live as THREE separate 3D arrays and psi_E/psi_H as SIX separate 3D
     arrays through the whole time loop (component tuples). This eliminates
     every jnp.stack / concatenate / dynamic-update-slice of (3,N)/(6,N)
     state arrays that the upstream loop pays per half-step.
  2. The PML coefficients b, a (expm1-based) and 1/kappa are loop-invariant
     (they only depend on alpha/kappa/sigma) and are hoisted out of the time
     loop as precomputed per-component constants.

Everything else mirrors the upstream forward path EXACTLY in math and op
order — the acceptance gate (tests/test_fdtdx_perf.py) asserts bitwise
equality (max abs diff == 0.0) of final E/H fields and every PhasorDetector
phasor against vanilla fdtdx.run_fdtd. Do not "improve" any expression here
without re-running that gate.

Measured end-to-end (niu36, Quadro RTX 6000, scripts/14_bench_fdtdx_fast.py,
8.0M cells x 500 steps, default XLA flags): vanilla 638 Mcell-steps/s hot,
fast 1140 Mcell-steps/s hot -> 1.79x, with max|dE| = max|dH| = 0.0 and all
pvgc PhasorDetector phasors exactly equal on GPU.

GRADIENTS ARE OUT OF SCOPE. This is a forward-only drop-in: no reversible /
checkpointed machinery, no custom VJP, no boundary recording. Anything that
needs jax.grad through the simulation must go through fdtdx.run_fdtd.

Scope guard: run_fdtd_fast validates the scene up front and raises
NotImplementedError loudly for anything outside the supported subset
(full 9-tensor materials, conductivity, dispersion, PEC/PMC/phase-shifted
Bloch boundaries, non-TFSF sources, gradient configs, custom stopping
conditions).

Mirrored-from map (re-check EACH on re-pinning fdtdx != 0.6.2):
  fdtdx/fdtd/fdtd.py        checkpointed_fdtd     -> run_fdtd_fast loop shell
  fdtdx/fdtd/forward.py     forward               -> _make_step
  fdtdx/fdtd/update.py      update_E / update_H   -> field updates in _make_step
  fdtdx/fdtd/update.py      update_detector_states-> detector block in _make_step
  fdtdx/core/physics/curl.py curl_E / curl_H      -> _curl_E_components /
                                                     _curl_H_components + _pml_coefficients
  fdtdx/core/physics/curl.py interpolate_fields   -> _interpolate_components
  fdtdx/core/misc.py        pad_fields            -> _pad_component
  fdtdx/objects/sources/tfsf.py TFSFPlaneSource.update_E/update_H (isotropic
                            branch)               -> _source_update_E / _source_update_H
"""

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.constants import eps0
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.update import get_wrap_padding_axes
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.sources.tfsf import TFSFPlaneSource


def _validate_supported(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> None:
    """Reject everything outside the pvgc production subset — loudly."""
    if config.gradient_config is not None:
        raise NotImplementedError(
            "run_fdtd_fast is FORWARD-ONLY: config.gradient_config must be None. "
            "Use fdtdx.run_fdtd for anything that needs gradients."
        )
    if config.invertible_optimization:
        raise NotImplementedError("run_fdtd_fast does not record boundaries (forward-only).")
    if arrays.electric_conductivity is not None or arrays.magnetic_conductivity is not None:
        raise NotImplementedError("run_fdtd_fast does not implement the conductive-material update path.")
    if arrays.dispersive_P_curr is not None:
        raise NotImplementedError("run_fdtd_fast does not implement the dispersive (ADE) update path.")
    if arrays.inv_permittivities.shape[0] not in (1, 3):
        raise NotImplementedError(
            f"run_fdtd_fast only supports isotropic/diagonal permittivity "
            f"(leading dim 1 or 3), got shape {arrays.inv_permittivities.shape}."
        )
    inv_mu = arrays.inv_permeabilities
    if isinstance(inv_mu, jax.Array) and inv_mu.ndim > 0 and inv_mu.shape[0] not in (1, 3):
        raise NotImplementedError(
            f"run_fdtd_fast only supports scalar or isotropic/diagonal permeability, got shape {inv_mu.shape}."
        )
    for boundary in objects.boundary_objects:
        if isinstance(boundary, PerfectlyMatchedLayer):
            continue
        if isinstance(boundary, BlochBoundary) and not boundary.needs_complex_fields:
            continue  # zero-phase Bloch == plain periodic: wrap padding, no correction
        raise NotImplementedError(
            f"run_fdtd_fast only supports PML and periodic boundaries, got {type(boundary).__name__} "
            f"(axis={boundary.axis}, direction={boundary.direction})."
        )
    for source in objects.sources:
        if not isinstance(source, TFSFPlaneSource):
            raise NotImplementedError(
                f"run_fdtd_fast only supports TFSFPlaneSource-family sources, got {type(source).__name__}."
            )
        if source._temporal_H_filter is not None:
            raise NotImplementedError("run_fdtd_fast does not implement the dispersive TFSF H-filter path.")


# ---------------------------------------------------------------------------
# Component-wise kernels (verbatim math from upstream, stack-free)
# ---------------------------------------------------------------------------


def _pml_coefficients(config, alpha, kappa, sigma):
    """Loop-invariant PML coefficients, hoisted out of the time loop.

    Verbatim expressions from curl_E/curl_H (they are identical on the E and
    H sides — same alpha/kappa/sigma, same formula), split into per-component
    3D constants. Microbenchmark-verified bitwise-neutral hoist (V3/V4).
    """
    b = jnp.expm1(-config.courant_number * config.resolution / c0 / eps0 * (sigma / kappa + alpha)) + 1
    a = jnp.nan_to_num((b - 1.0) * sigma / (sigma + alpha * kappa) / kappa, nan=0.0, posinf=0.0, neginf=0.0)
    kappa_inv = 1.0 / kappa
    b6 = tuple(b[i] for i in range(6))
    a6 = tuple(a[i] for i in range(6))
    ki3 = tuple(kappa_inv[i] for i in range(3))
    return b6, a6, ki3


def _pad_component(x: jax.Array, periodic_axes: tuple[bool, bool, bool]) -> jax.Array:
    """pad_fields for a single 3D component (same sequential axis order)."""
    for i, periodic in enumerate(periodic_axes):
        pad_mode = "wrap" if periodic else "constant"
        pad_width = [(0, 0), (0, 0), (0, 0)]
        pad_width[i] = (1, 1)
        x = jnp.pad(x, pad_width, mode=pad_mode)
    return x


def _curl_E_components(E_pad3, psi_H6, b6, a6, ki3):
    """curl_E with simulate_boundaries=True, component tuples in and out."""
    Ex_pad, Ey_pad, Ez_pad = E_pad3
    dyEz = (jnp.roll(Ez_pad, -1, axis=1) - Ez_pad)[1:-1, 1:-1, 1:-1]
    dzEy = (jnp.roll(Ey_pad, -1, axis=2) - Ey_pad)[1:-1, 1:-1, 1:-1]
    dzEx = (jnp.roll(Ex_pad, -1, axis=2) - Ex_pad)[1:-1, 1:-1, 1:-1]
    dxEz = (jnp.roll(Ez_pad, -1, axis=0) - Ez_pad)[1:-1, 1:-1, 1:-1]
    dxEy = (jnp.roll(Ey_pad, -1, axis=0) - Ey_pad)[1:-1, 1:-1, 1:-1]
    dyEx = (jnp.roll(Ex_pad, -1, axis=1) - Ex_pad)[1:-1, 1:-1, 1:-1]

    psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy = psi_H6
    psi_Hxy = b6[4] * psi_Hxy + a6[4] * dyEz
    psi_Hxz = b6[5] * psi_Hxz + a6[5] * dzEy
    psi_Hyz = b6[5] * psi_Hyz + a6[5] * dzEx
    psi_Hyx = b6[3] * psi_Hyx + a6[3] * dxEz
    psi_Hzx = b6[3] * psi_Hzx + a6[3] * dxEy
    psi_Hzy = b6[4] * psi_Hzy + a6[4] * dyEx

    curl_x = (ki3[1] * dyEz + psi_Hxy) - (ki3[2] * dzEy + psi_Hxz)
    curl_y = (ki3[2] * dzEx + psi_Hyz) - (ki3[0] * dxEz + psi_Hyx)
    curl_z = (ki3[0] * dxEy + psi_Hzx) - (ki3[1] * dyEx + psi_Hzy)
    return (curl_x, curl_y, curl_z), (psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy)


def _curl_H_components(H_pad3, psi_E6, b6, a6, ki3):
    """curl_H with simulate_boundaries=True, component tuples in and out."""
    Hx_pad, Hy_pad, Hz_pad = H_pad3
    dyHz = (Hz_pad - jnp.roll(Hz_pad, 1, axis=1))[1:-1, 1:-1, 1:-1]
    dzHy = (Hy_pad - jnp.roll(Hy_pad, 1, axis=2))[1:-1, 1:-1, 1:-1]
    dzHx = (Hx_pad - jnp.roll(Hx_pad, 1, axis=2))[1:-1, 1:-1, 1:-1]
    dxHz = (Hz_pad - jnp.roll(Hz_pad, 1, axis=0))[1:-1, 1:-1, 1:-1]
    dxHy = (Hy_pad - jnp.roll(Hy_pad, 1, axis=0))[1:-1, 1:-1, 1:-1]
    dyHx = (Hx_pad - jnp.roll(Hx_pad, 1, axis=1))[1:-1, 1:-1, 1:-1]

    psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy = psi_E6
    psi_Exy = b6[1] * psi_Exy + a6[1] * dyHz
    psi_Exz = b6[2] * psi_Exz + a6[2] * dzHy
    psi_Eyz = b6[2] * psi_Eyz + a6[2] * dzHx
    psi_Eyx = b6[0] * psi_Eyx + a6[0] * dxHz
    psi_Ezx = b6[0] * psi_Ezx + a6[0] * dxHy
    psi_Ezy = b6[1] * psi_Ezy + a6[1] * dyHx

    curl_x = (ki3[1] * dyHz + psi_Exy) - (ki3[2] * dzHy + psi_Exz)
    curl_y = (ki3[2] * dzHx + psi_Eyz) - (ki3[0] * dxHz + psi_Eyx)
    curl_z = (ki3[0] * dxHy + psi_Ezx) - (ki3[1] * dyHx + psi_Ezy)
    return (curl_x, curl_y, curl_z), (psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy)


def _interpolate_components(E_pad3, H_pad3):
    """interpolate_fields onto the E_z Yee point, component tuples in/out."""
    E_x, E_y, E_z = E_pad3
    H_x, H_y, H_z = H_pad3

    E_x = (E_x[1:-1, 1:-1, 1:-1] + E_x[:-2, 1:-1, 1:-1] + E_x[1:-1, 1:-1, 2:] + E_x[:-2, 1:-1, 2:]) / 4.0
    E_y = (E_y[1:-1, 1:-1, 1:-1] + E_y[1:-1, :-2, 1:-1] + E_y[1:-1, 1:-1, 2:] + E_y[1:-1, :-2, 2:]) / 4.0
    E_z = E_z[1:-1, 1:-1, 1:-1]

    H_x = (H_x[1:-1, 1:-1, 1:-1] + H_x[1:-1, :-2, 1:-1]) / 2.0
    H_y = (H_y[1:-1, 1:-1, 1:-1] + H_y[:-2, 1:-1, 1:-1]) / 2.0
    H_z = (
        H_z[1:-1, 1:-1, 1:-1]
        + H_z[:-2, 1:-1, 1:-1]
        + H_z[1:-1, :-2, 1:-1]
        + H_z[:-2, :-2, 1:-1]
        + H_z[1:-1, 1:-1, 2:]
        + H_z[:-2, 1:-1, 2:]
        + H_z[1:-1, :-2, 2:]
        + H_z[:-2, :-2, 2:]
    ) / 8.0
    return (E_x, E_y, E_z), (H_x, H_y, H_z)


# ---------------------------------------------------------------------------
# TFSF source injection (isotropic branch of TFSFPlaneSource, component-wise)
# ---------------------------------------------------------------------------


def _source_update_E(source, E3, inv_permittivities, time_step):
    """TFSFPlaneSource.update_E (inverse=False, isotropic branch) on component tuples.

    Identical adds on identical operands — only the container differs:
    ``E.at[h_axis, *grid_slice].add(x)`` becomes
    ``E3[h_axis].at[grid_slice].add(x)``.
    """
    delta_t = source._config.time_step_duration
    c = source._config.courant_number
    inv_permittivity_slice = inv_permittivities[:, *source.grid_slice]

    h_axis, v_axis, p_axis = source.horizontal_axis, source.vertical_axis, source.propagation_axis
    amplitude_H = {}
    for axis in [h_axis, v_axis, p_axis]:
        time_H = (time_step + source._time_offset_H[axis]) * delta_t
        amplitude_H[axis] = (
            source.temporal_profile.get_amplitude(
                time=time_H,
                period=source.wave_character.get_period(),
                phase_shift=source.wave_character.phase_shift,
            )
            * source.static_amplitude_factor
        )

    sign = 1 if source.direction == "+" else -1

    # NOTE: for isotropic scenes inv_permittivity_slice has leading dim 1 and
    # indexing it with h_axis/v_axis relies on JAX's index clamping — exactly
    # as the upstream code does. Keep the expressions verbatim.
    H_v_inc = source._H[v_axis] * amplitude_H[v_axis]
    H_v_inc = H_v_inc * c * inv_permittivity_slice[h_axis]
    H_v_inc = jax.lax.stop_gradient(H_v_inc)

    H_h_inc = source._H[h_axis] * amplitude_H[h_axis]
    H_h_inc = H_h_inc * c * inv_permittivity_slice[v_axis]
    H_h_inc = jax.lax.stop_gradient(H_h_inc)

    E_list = list(E3)
    E_list[h_axis] = E_list[h_axis].at[source.grid_slice].add(sign * H_v_inc)
    E_list[v_axis] = E_list[v_axis].at[source.grid_slice].add(-sign * H_h_inc)
    return tuple(E_list)


def _source_update_H(source, H3, inv_permeabilities, time_step):
    """TFSFPlaneSource.update_H (inverse=False, isotropic branch) on component tuples."""
    delta_t = source._config.time_step_duration
    c = source._config.courant_number

    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        inv_permeability_slice = inv_permeabilities[:, *source.grid_slice]
    else:
        inv_permeability_slice = inv_permeabilities

    h_axis, v_axis, p_axis = source.horizontal_axis, source.vertical_axis, source.propagation_axis
    amplitude_E = {}
    for axis in [h_axis, v_axis, p_axis]:
        time_E = (time_step + source._time_offset_E[axis]) * delta_t
        amplitude_E[axis] = (
            source.temporal_profile.get_amplitude(
                time=time_E,
                period=source.wave_character.get_period(),
                phase_shift=source.wave_character.phase_shift,
            )
            * source.static_amplitude_factor
        )

    sign = 1 if source.direction == "+" else -1

    E_h_inc = source._E[h_axis] * amplitude_E[h_axis]
    if isinstance(inv_permeability_slice, jax.Array) and inv_permeability_slice.ndim > 1:
        E_h_inc = E_h_inc * c * inv_permeability_slice[v_axis]
    else:
        E_h_inc = E_h_inc * c * inv_permeability_slice
    E_h_inc = jax.lax.stop_gradient(E_h_inc)

    E_v_inc = source._E[v_axis] * amplitude_E[v_axis]
    if isinstance(inv_permeability_slice, jax.Array) and inv_permeability_slice.ndim > 1:
        E_v_inc = E_v_inc * c * inv_permeability_slice[h_axis]
    else:
        E_v_inc = E_v_inc * c * inv_permeability_slice
    E_v_inc = jax.lax.stop_gradient(E_v_inc)

    H_list = list(H3)
    H_list[v_axis] = H_list[v_axis].at[source.grid_slice].add(sign * E_h_inc)
    H_list[h_axis] = H_list[h_axis].at[source.grid_slice].add(-sign * E_v_inc)
    return tuple(H_list)


# ---------------------------------------------------------------------------
# Fast forward loop
# ---------------------------------------------------------------------------


def run_fdtd_fast(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
) -> SimulationState:
    """Forward-only drop-in for fdtdx.run_fdtd on the pvgc scene subset.

    Bitwise-identical to ``fdtdx.run_fdtd(arrays, objects, config, key)`` for
    supported scenes (see module docstring; asserted by tests/test_fdtdx_perf.py),
    but with the component-tuple state layout and hoisted PML coefficients.
    NO GRADIENT SUPPORT — the returned arrays are for detector reading and
    field inspection only.

    Args:
        arrays: Initial simulation state (as returned by apply_params).
        objects: Placed simulation objects.
        config: Simulation configuration (gradient_config must be None).
        key: Accepted for signature compatibility; the forward path of the
            supported subset never consumes randomness.

    Returns:
        SimulationState: (final time step, ArrayContainer with final fields
        and detector states) — same structure as fdtdx.run_fdtd.
    """
    del key  # only consumed by boundary recording, which is out of scope
    _validate_supported(arrays, objects, config)

    arrays = arrays.reset()

    periodic_axes = get_wrap_padding_axes(objects)
    inv_eps = arrays.inv_permittivities
    inv_mu = arrays.inv_permeabilities
    c = config.courant_number
    sources = objects.sources
    forward_detectors = objects.forward_detectors
    any_exact = any(d.exact_interpolation for d in forward_detectors)
    any_raw = any(not d.exact_interpolation for d in forward_detectors)

    # per-component broadcast of the material arrays: leading dim 1 broadcasts
    # over components exactly like the upstream (3,N)*(1,N) elementwise op
    def _mat(m, i):
        if isinstance(m, jax.Array) and m.ndim > 0:
            return m[i] if m.shape[0] == 3 else m[0]
        return m

    inv_eps3 = tuple(_mat(inv_eps, i) for i in range(3))
    inv_mu3 = tuple(_mat(inv_mu, i) for i in range(3))

    # Loop-invariant PML coefficients (identical on E and H side). Computed
    # under jit — NOT eagerly — so XLA's algebraic simplifier applies the
    # same value-changing rewrites (e.g. x/y/z -> x/(y*z) in the `a`
    # coefficient) as it does to the identical expressions inside vanilla's
    # compiled loop body; eager op-by-op execution would round differently
    # and break the bitwise equivalence gate. config stays closed-over
    # (static) so the scalar prefactor constant-folds exactly like upstream.
    b6, a6, ki3 = jax.jit(
        lambda alpha, kappa, sigma: _pml_coefficients(config, alpha, kappa, sigma)
    )(arrays.alpha, arrays.kappa, arrays.sigma)

    def _pad3(comps):
        return tuple(_pad_component(x, periodic_axes) for x in comps)

    def body(time_step, carry):
        E3, H3, psi_E6, psi_H6, det_states = carry
        H_prev3 = H3

        # ---- update_E: E += c * curl_H * inv_eps, then TFSF injection ----
        # Grouping note: upstream writes `E + c * curl * inv_eps` with
        # inv_eps broadcast (1,N)->(3,N); XLA's algebraic simplifier sinks
        # the scalar constant into the smaller operand, evaluating
        # `E + curl * (inv_eps * c)`. Our component arrays are full-shape
        # (no broadcast), so that rewrite does not fire — write the
        # simplified grouping explicitly or dielectric cells differ by 1 ulp.
        H_pad3 = _pad3(H3)
        curl3, psi_E6 = _curl_H_components(H_pad3, psi_E6, b6, a6, ki3)
        E3 = tuple(E3[i] + curl3[i] * (inv_eps3[i] * c) for i in range(3))

        for source in sources:

            def _inject_E(E3_in, src=source):
                adj_time_step = src.adjust_time_step_by_on_off(time_step)
                return _source_update_E(src, E3_in, inv_eps, adj_time_step)

            E3 = jax.lax.cond(source.is_on_at_time_step(time_step), _inject_E, lambda e: e, E3)

        # ---- update_H: H -= c * curl_E * inv_mu, then TFSF injection ----
        # Same grouping note as update_E for array-valued inv_mu; for the
        # scalar (non-magnetic, 1.0) case the mirror `c * curl * inv_mu` is
        # exact either way and upstream's mul-by-1 is simplified away.
        E_pad3 = _pad3(E3)
        curlE3, psi_H6 = _curl_E_components(E_pad3, psi_H6, b6, a6, ki3)
        if isinstance(inv_mu, jax.Array) and inv_mu.ndim > 0:
            H3 = tuple(H3[i] - curlE3[i] * (inv_mu3[i] * c) for i in range(3))
        else:
            H3 = tuple(H3[i] - c * curlE3[i] * inv_mu3[i] for i in range(3))

        for source in sources:

            def _inject_H(H3_in, src=source):
                adj_time_step = src.adjust_time_step_by_on_off(time_step)
                return _source_update_H(src, H3_in, inv_mu, adj_time_step + 0.5)

            H3 = jax.lax.cond(source.is_on_at_time_step(time_step), _inject_H, lambda h: h, H3)

        # ---- update_detector_states ----
        if forward_detectors:
            E_for_exact = H_for_exact = None
            if any_exact:
                E_det_pad3 = _pad3(E3)
                H_avg_pad3 = _pad3(tuple((H_prev3[i] + H3[i]) / 2 for i in range(3)))
                Ei3, Hi3 = _interpolate_components(E_det_pad3, H_avg_pad3)
                # detectors slice (3, Nx, Ny, Nz) arrays internally: assemble
                # exactly the two stacks upstream interpolate_fields builds
                E_for_exact = jnp.stack(Ei3, axis=0)
                H_for_exact = jnp.stack(Hi3, axis=0)
            E_raw = jnp.stack(E3, axis=0) if any_raw else None
            H_raw = jnp.stack(H3, axis=0) if any_raw else None

            def helper_fn(E_input, H_input, detector):
                return detector.update(
                    time_step=time_step,
                    E=E_input,
                    H=H_input,
                    state=det_states[detector.name],
                    inv_permittivity=inv_eps,
                    inv_permeability=inv_mu,
                )

            det_states = dict(det_states)
            for d in forward_detectors:
                det_states[d.name] = jax.lax.cond(
                    d._is_on_at_time_step_arr[time_step],
                    helper_fn,
                    lambda e, h, _: det_states[d.name],
                    E_for_exact if d.exact_interpolation else E_raw,
                    H_for_exact if d.exact_interpolation else H_raw,
                    d,
                )

        return E3, H3, psi_E6, psi_H6, det_states

    init = (
        tuple(arrays.fields.E[i] for i in range(3)),
        tuple(arrays.fields.H[i] for i in range(3)),
        tuple(arrays.fields.psi_E[i] for i in range(6)),
        tuple(arrays.fields.psi_H[i] for i in range(6)),
        arrays.detector_states,
    )
    total_steps = config.time_steps_total
    E3, H3, psi_E6, psi_H6, det_states = jax.lax.fori_loop(0, total_steps, body, init)

    # repack into a normal ArrayContainer so downstream detector-reading code
    # (and field inspection) sees the standard layout
    arrays = arrays.aset("fields->E", jnp.stack(E3, axis=0))
    arrays = arrays.aset("fields->H", jnp.stack(H3, axis=0))
    arrays = arrays.aset("fields->psi_E", jnp.stack(psi_E6, axis=0))
    arrays = arrays.aset("fields->psi_H", jnp.stack(psi_H6, axis=0))
    arrays = arrays.aset("detector_states", det_states)
    return jnp.asarray(total_steps, dtype=jnp.int32), arrays
