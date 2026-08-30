"""Workload-specialized FORWARD-ONLY fast path for fdtdx.run_fdtd.

Vendors the inner time loop of fdtdx (c) Yannik Mahlau et al., MIT License
(github.com/ymahlau/fdtdx) (the sanctioned "vendor one module when blocked"
escape hatch — see README Layer A notes). The upstream loop is written for
generality (anisotropic tensors, conductivity, dispersion, reversible
gradients); the production scenes in this repo only ever exercise the
isotropic + PML/periodic + TFSF-plane-source + PhasorDetector subset, so this
module specializes the hot loop for that subset and keeps the general engine
as the reference. Worth offering upstream once generalized.

What it does differently (validated by scripts/13_curl_microbench.py on a
Turing-class GPU, 16.8M cells, ~1.9x kernel speedup, bitwise-identical output;
slab variant V5 re-validated bitwise at 256^3/200^3 with npml 16-32):

  1. E/H live as THREE separate 3D arrays through the whole time loop
     (component tuples). This eliminates every jnp.stack / concatenate /
     dynamic-update-slice of (3,N) state arrays that the upstream loop pays
     per half-step.
  2. The PML coefficients b, a (expm1-based) and 1/kappa are loop-invariant
     (they only depend on alpha/kappa/sigma) and are hoisted out of the time
     loop as precomputed constants.
  3. CPML state and coefficients live only on the PML slabs. With the
     upstream defaults the base arrays are kappa=1, sigma=0, alpha=0 and the
     PML objects write their graded profiles only inside their grid_slice
     (rows axis and axis+3), so outside every slab b=1 and a=0 exactly and
     psi (starting from 0) stays exactly 0 forever — verified empirically on
     placed arrays in tests/test_fdtdx_perf.py. The interior curl is then
     the plain Yee difference (kappa=1: the 1/kappa multiply drops), and
     each psi component plus its b/a/1-kappa coefficients are stored as
     slab-sized arrays only. Exactness in slab corners comes from
     factorizing the curl per derivative axis: each factor
     T = (1/kappa)*d + psi involves only ONE axis's coefficients, so
     per-axis slab .at[slice].set updates compose exactly.

  Slab-variant routing data (scripts/13 --slab, Quadro RTX 6000, donated
  carry, H-half-step kernel, speedups vs the verbatim V0): 3-D slab
  coefficient arrays (V5) track the per-axis slab fraction 2*npml/n and
  can LOSE to V4 at high fractions (2.71x at 12%, but 1.71x-vs-V4's-1.91x
  at the misaligned 20%). Storing the coefficients as 1-D depth profiles
  broadcast along each face (V6 — CPML coefficients are functions of PML
  depth only: Roden & Gedney; gprMax/flaport-fdtd/Schneider ch.11 store 1-D
  vectors; B-CALM is the prior art for keeping the recursive accumulators
  only in the PML regions) wins at EVERY geometry tested: 3.08x @ 256^3
  npml16, 2.00x @ 256^3 npml25, 2.09x @ 200^3 npml25, 2.57x @ 200^3 npml16
  (V4 = 1.79-1.94x on the same grids). All bitwise-identical to the
  verbatim kernel. Adopted: V6. The state memory drops from 30 full-volume
  scalar fields (12 psi + 18 alpha/kappa/sigma) to slab psi + 1-D
  coefficient vectors — the capacity win is unconditional.

Everything else mirrors the upstream forward path EXACTLY in math and op
order — the acceptance gate (tests/test_fdtdx_perf.py) asserts bitwise
equality (max abs diff == 0.0) of final E/H fields and every PhasorDetector
phasor against vanilla fdtdx.run_fdtd. Do not "improve" any expression here
without re-running that gate.

Measured end-to-end (Quadro RTX 6000, scripts/14_bench_fdtdx_fast.py,
8.0M cells x 500 steps, default XLA flags, per-engine processes):
  vanilla          640 Mcell-steps/s hot   peak 6.02 GiB
  fast (slab V6)  1245 Mcell-steps/s hot   peak 3.07 GiB   -> 1.94x
  fast+reclaim    1163 Mcell-steps/s hot   peak 2.09 GiB   -> 1.82x
with max|dE| = max|dH| = 0.0 and all grating_coupler PhasorDetector phasors exactly
equal on GPU (both scripts/14 and the real grating_coupler._run dispatch path).
At the low-slab-fraction geometry (16.8M cells, 256^3, npml16, 300 steps):
vanilla 653 -> fast 1529 Mcell-steps/s, 2.34x, max|dE|=max|dH|=0.0.
(The earlier V4 component-tuple loop measured 1140 Mcell-steps/s = 1.79x on
the 8.0M case; the slab restriction added the rest plus the memory drop.)

Capacity on the 23GiB-usable Turing (steps=100 probes): vanilla runs 15.6M
cells (peak 11.86 GiB) and OOMs in the loop at 27M; the fast loop runs 27M
without reclaim (peak 10.16 GiB) and with reclaim_memory=True runs 44.7M
(peak 10.69 GiB) and 64M cells (peak 15.12 GiB). Beyond that (77M) the
ceiling is fdtdx's own SETUP (place_objects allocates the full-volume
alpha/kappa/sigma/psi before this loop ever runs), not the time loop —
lifting that requires touching scene initialization, out of scope here.

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
from fdtdx.fdtd.container import ArrayContainer, FieldState, ObjectContainer, SimulationState
from fdtdx.fdtd.update import get_wrap_padding_axes
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.sources.tfsf import TFSFPlaneSource


def _validate_supported(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> None:
    """Reject everything outside the grating_coupler production subset — loudly."""
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
    """Loop-invariant PML coefficients from alpha/kappa/sigma slices.

    Verbatim expressions from curl_E/curl_H (they are identical on the E and
    H sides — same formula on different coefficient rows). Works on arrays
    of any shape (full rows or slab slices). Microbenchmark-verified
    bitwise-neutral hoist (V3/V4/V5).
    """
    b = jnp.expm1(-config.courant_number * config.resolution / c0 / eps0 * (sigma / kappa + alpha)) + 1
    a = jnp.nan_to_num((b - 1.0) * sigma / (sigma + alpha * kappa) / kappa, nan=0.0, posinf=0.0, neginf=0.0)
    kappa_inv = 1.0 / kappa
    return b, a, kappa_inv


# psi component ordering as unpacked in curl_E/curl_H: (xy, xz, yz, yx, zx, zy)
# — the second letter is the derivative axis each component convolves.
_PSI_DERIV_AXES = (1, 2, 2, 0, 0, 1)


def _collect_pml_slabs(objects: ObjectContainer) -> dict[int, list[tuple[slice, slice, slice]]]:
    """Static slab regions (grid_slice) of the placed PML objects, per axis.

    The PML profiles (and therefore all nonzero psi/b/a and kappa != 1) live
    exactly inside these regions — coefficient row d (and d+3) is modified
    only by PMLs with axis == d (PerfectlyMatchedLayer.modify_arrays).
    """
    slabs: dict[int, list[tuple[slice, slice, slice]]] = {0: [], 1: [], 2: []}
    for pml in objects.pml_objects:
        slabs[pml.axis].append(pml.grid_slice)
    for axis, slab_list in slabs.items():
        spans = sorted((s[axis].start, s[axis].stop) for s in slab_list)
        for (lo1, hi1), (lo2, hi2) in zip(spans, spans[1:]):
            if hi1 > lo2:
                raise NotImplementedError(
                    f"run_fdtd_fast: overlapping PML slabs on axis {axis} "
                    f"({spans}) — slab-restricted psi state would double-count."
                )
    return slabs


def _slab_coefficients(config, alpha, kappa, sigma, slabs):
    """Slab loop-invariant PML coefficients as 1-D depth profiles.

    Returns per-axis lists of (bE, aE, bH, aH, ki) arrays of broadcastable
    shape (npml along the face normal, 1 transversally): bE/aE from
    coefficient row d (used by the psi_E updates in curl_H), bH/aH from row
    d+3 (psi_H updates in curl_E), ki = 1/kappa from row d (upstream uses
    kappa rows 0..2 in BOTH curls).

    CPML coefficients are functions of PML depth only (Roden & Gedney;
    gprMax, flaport/fdtd and Schneider ch.11 all store 1-D vectors), and
    fdtdx's modify_arrays builds each face's alpha/kappa/sigma by
    broadcasting a 1-D polynomial profile — verified exactly on placed
    arrays, and guarded here: a transversely varying profile (custom PML
    subclass) raises instead of being silently averaged. Microbenchmark V6:
    the 1-D form beats 3-D slab coefficients at every geometry tested and
    stays bitwise-identical.

    Computed under jit — NOT eagerly — so XLA's algebraic simplifier applies
    the same value-changing rewrites (e.g. x/y/z -> x/(y*z) in the `a`
    coefficient) as it does to the identical expressions inside vanilla's
    compiled loop body. config stays closed-over (static) so the scalar
    prefactor constant-folds exactly like upstream.
    """
    profiles = {}
    for d, slab_list in slabs.items():
        rows = []
        for sl in slab_list:
            idx_1d = tuple(slice(None) if i == d else slice(0, 1) for i in range(3))
            per_row = []
            for row in (d, d + 3):
                triple = []
                for name, arr in (("alpha", alpha), ("kappa", kappa), ("sigma", sigma)):
                    block = arr[row][sl]
                    prof = block[idx_1d]
                    if not bool(jnp.all(block == prof)):
                        raise NotImplementedError(
                            f"run_fdtd_fast: {name} row {row} varies transversely "
                            f"inside the axis-{d} PML slab {sl} — the 1-D depth-"
                            f"profile coefficient layout does not apply."
                        )
                    triple.append(prof)
                per_row.append(tuple(triple))
            rows.append(tuple(per_row))
        profiles[d] = rows

    def compute(prof_tree):
        out = {}
        for d, slab_rows in prof_tree.items():
            rows = []
            for (alE, kaE, siE), (alH, kaH, siH) in slab_rows:
                bE, aE, _ = _pml_coefficients(config, alE, kaE, siE)
                bH, aH, _ = _pml_coefficients(config, alH, kaH, siH)
                ki = 1.0 / kaE
                rows.append((bE, aE, bH, aH, ki))
            out[d] = rows
        return out

    return jax.jit(compute)(profiles)


def _slab_factor(diff, slab_list, psi_list, coeff_list, side):
    """One factorized curl term: T = diff outside the slabs of this axis,
    (1/kappa)*diff + psi_new inside; psi_new = b*psi + a*diff (slab-sized).

    Elementwise identical to upstream's full-volume ki*diff + psi term: the
    only outside-slab deviation is dropping the exact *1 and +0, and slab
    corners are exact because the OTHER curl term's coefficients live on a
    different axis's slabs (per-axis factorization).

    side selects the coefficient rows: "E" for psi_E updates (curl_H, rows
    0..2), "H" for psi_H updates (curl_E, rows 3..5).
    """
    new_psi = []
    T = diff
    for sl, psi_s, (bE, aE, bH, aH, ki) in zip(slab_list, psi_list, coeff_list):
        b_s, a_s = (bE, aE) if side == "E" else (bH, aH)
        d_s = diff[sl]
        p_new = b_s * psi_s + a_s * d_s
        T = T.at[sl].set(ki * d_s + p_new)
        new_psi.append(p_new)
    return T, tuple(new_psi)


def _pad_component(x: jax.Array, periodic_axes: tuple[bool, bool, bool]) -> jax.Array:
    """pad_fields for a single 3D component (same sequential axis order)."""
    for i, periodic in enumerate(periodic_axes):
        pad_mode = "wrap" if periodic else "constant"
        pad_width = [(0, 0), (0, 0), (0, 0)]
        pad_width[i] = (1, 1)
        x = jnp.pad(x, pad_width, mode=pad_mode)
    return x


def _curl_E_components(E_pad3, psi_H6, slabs, coeffs):
    """curl_E with simulate_boundaries=True: component tuples, slab psi state.

    psi_H6 is a 6-tuple (xy, xz, yz, yx, zx, zy) of per-slab array tuples
    (empty for axes without PML — those components are identically zero).
    """
    Ex_pad, Ey_pad, Ez_pad = E_pad3
    dyEz = (jnp.roll(Ez_pad, -1, axis=1) - Ez_pad)[1:-1, 1:-1, 1:-1]
    dzEy = (jnp.roll(Ey_pad, -1, axis=2) - Ey_pad)[1:-1, 1:-1, 1:-1]
    dzEx = (jnp.roll(Ex_pad, -1, axis=2) - Ex_pad)[1:-1, 1:-1, 1:-1]
    dxEz = (jnp.roll(Ez_pad, -1, axis=0) - Ez_pad)[1:-1, 1:-1, 1:-1]
    dxEy = (jnp.roll(Ey_pad, -1, axis=0) - Ey_pad)[1:-1, 1:-1, 1:-1]
    dyEx = (jnp.roll(Ex_pad, -1, axis=1) - Ex_pad)[1:-1, 1:-1, 1:-1]

    p_xy, p_xz, p_yz, p_yx, p_zx, p_zy = psi_H6
    T1x, p_xy = _slab_factor(dyEz, slabs[1], p_xy, coeffs[1], "H")
    T2x, p_xz = _slab_factor(dzEy, slabs[2], p_xz, coeffs[2], "H")
    T1y, p_yz = _slab_factor(dzEx, slabs[2], p_yz, coeffs[2], "H")
    T2y, p_yx = _slab_factor(dxEz, slabs[0], p_yx, coeffs[0], "H")
    T1z, p_zx = _slab_factor(dxEy, slabs[0], p_zx, coeffs[0], "H")
    T2z, p_zy = _slab_factor(dyEx, slabs[1], p_zy, coeffs[1], "H")

    curl_x = T1x - T2x
    curl_y = T1y - T2y
    curl_z = T1z - T2z
    return (curl_x, curl_y, curl_z), (p_xy, p_xz, p_yz, p_yx, p_zx, p_zy)


def _curl_H_components(H_pad3, psi_E6, slabs, coeffs):
    """curl_H with simulate_boundaries=True: component tuples, slab psi state."""
    Hx_pad, Hy_pad, Hz_pad = H_pad3
    dyHz = (Hz_pad - jnp.roll(Hz_pad, 1, axis=1))[1:-1, 1:-1, 1:-1]
    dzHy = (Hy_pad - jnp.roll(Hy_pad, 1, axis=2))[1:-1, 1:-1, 1:-1]
    dzHx = (Hx_pad - jnp.roll(Hx_pad, 1, axis=2))[1:-1, 1:-1, 1:-1]
    dxHz = (Hz_pad - jnp.roll(Hz_pad, 1, axis=0))[1:-1, 1:-1, 1:-1]
    dxHy = (Hy_pad - jnp.roll(Hy_pad, 1, axis=0))[1:-1, 1:-1, 1:-1]
    dyHx = (Hx_pad - jnp.roll(Hx_pad, 1, axis=1))[1:-1, 1:-1, 1:-1]

    p_xy, p_xz, p_yz, p_yx, p_zx, p_zy = psi_E6
    T1x, p_xy = _slab_factor(dyHz, slabs[1], p_xy, coeffs[1], "E")
    T2x, p_xz = _slab_factor(dzHy, slabs[2], p_xz, coeffs[2], "E")
    T1y, p_yz = _slab_factor(dzHx, slabs[2], p_yz, coeffs[2], "E")
    T2y, p_yx = _slab_factor(dxHz, slabs[0], p_yx, coeffs[0], "E")
    T1z, p_zx = _slab_factor(dxHy, slabs[0], p_zx, coeffs[0], "E")
    T2z, p_zy = _slab_factor(dyHx, slabs[1], p_zy, coeffs[1], "E")

    curl_x = T1x - T2x
    curl_y = T1y - T2y
    curl_z = T1z - T2z
    return (curl_x, curl_y, curl_z), (p_xy, p_xz, p_yz, p_yx, p_zx, p_zy)


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


def _scatter_psi(psi6, slabs, vol_shape, dtype):
    """Slab psi state -> full (6, *vol_shape) array (exact zeros outside)."""
    full = jnp.zeros((6, *vol_shape), dtype=dtype)
    for i, dax in enumerate(_PSI_DERIV_AXES):
        for sl, p in zip(slabs[dax], psi6[i]):
            full = full.at[(i, *sl)].set(p)
    return full


def _rebuild_pml_arrays(arrays_shape, dtype, objects):
    """Recompute alpha/kappa/sigma exactly as fdtdx initialization does.

    Used by reclaim_memory=True to restore the drop-in return contract after
    the full-volume originals were freed: base values (0 / 1 / 0) plus each
    boundary's modify_arrays profile — deterministic, so the rebuilt arrays
    equal the originals bit-for-bit (asserted in tests/test_fdtdx_perf.py).
    """
    alpha = jnp.zeros((6, *arrays_shape), dtype=dtype)
    kappa = jnp.ones((6, *arrays_shape), dtype=dtype)
    sigma = jnp.zeros((6, *arrays_shape), dtype=dtype)
    for boundary in objects.boundary_objects:
        modify_fn = getattr(boundary, "modify_arrays", None)
        if callable(modify_fn):
            result = modify_fn(
                alpha=alpha,
                kappa=kappa,
                sigma=sigma,
                electric_conductivity=None,
                magnetic_conductivity=None,
            )
            if result is not None:
                alpha = result.get("alpha", alpha)
                kappa = result.get("kappa", kappa)
                sigma = result.get("sigma", sigma)
    return alpha, kappa, sigma


def run_fdtd_fast(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    reclaim_memory: bool = False,
) -> SimulationState:
    """Forward-only drop-in for fdtdx.run_fdtd on the grating_coupler scene subset.

    Bitwise-identical to ``fdtdx.run_fdtd(arrays, objects, config, key)`` for
    supported scenes (see module docstring; asserted by tests/test_fdtdx_perf.py),
    but with the component-tuple state layout, hoisted PML coefficients and
    slab-restricted CPML state. NO GRADIENT SUPPORT — the returned arrays are
    for detector reading and field inspection only.

    Args:
        arrays: Initial simulation state (as returned by apply_params).
        objects: Placed simulation objects.
        config: Simulation configuration (gradient_config must be None).
        key: Accepted for signature compatibility; the forward path of the
            supported subset never consumes randomness.
        reclaim_memory: When True, FREES the caller's full-volume device
            buffers (fields.E/H, psi_E/psi_H, alpha/kappa/sigma) after the
            slab coefficients are extracted, so the time loop runs without
            30+ full-volume scalar fields resident — this is what unlocks
            grids that OOM vanilla. The INPUT ``arrays`` container is
            invalid afterwards (any use of its deleted leaves raises);
            the RETURNED container is complete and bit-identical (psi is
            re-scattered, alpha/kappa/sigma deterministically rebuilt).
            Only enable when the caller discards its input container, as
            invdx's grating_coupler runners do. Defaults to False.

    Returns:
        SimulationState: (final time step, ArrayContainer with final fields
        and detector states) — same structure as fdtdx.run_fdtd.
    """
    del key  # only consumed by boundary recording, which is out of scope
    _validate_supported(arrays, objects, config)

    periodic_axes = get_wrap_padding_axes(objects)
    inv_eps = arrays.inv_permittivities
    inv_mu = arrays.inv_permeabilities
    c = config.courant_number
    sources = objects.sources
    forward_detectors = objects.forward_detectors
    any_exact = any(d.exact_interpolation for d in forward_detectors)
    any_raw = any(not d.exact_interpolation for d in forward_detectors)
    vol_shape = arrays.fields.E.shape[1:]
    dtype = arrays.fields.E.dtype

    # per-component broadcast of the material arrays: leading dim 1 broadcasts
    # over components exactly like the upstream (3,N)*(1,N) elementwise op
    def _mat(m, i):
        if isinstance(m, jax.Array) and m.ndim > 0:
            return m[i] if m.shape[0] == 3 else m[0]
        return m

    inv_eps3 = tuple(_mat(inv_eps, i) for i in range(3))
    inv_mu3 = tuple(_mat(inv_mu, i) for i in range(3))

    # Slab-restricted loop-invariant PML coefficients (see _slab_coefficients
    # for the jit-not-eager bitwise rationale).
    slabs = _collect_pml_slabs(objects)
    coeffs = _slab_coefficients(config, arrays.alpha, arrays.kappa, arrays.sigma, slabs)

    if reclaim_memory:
        jax.block_until_ready(coeffs)
        for buf in (
            arrays.fields.E,
            arrays.fields.H,
            arrays.fields.psi_E,
            arrays.fields.psi_H,
            arrays.alpha,
            arrays.kappa,
            arrays.sigma,
        ):
            buf.delete()

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
        curl3, psi_E6 = _curl_H_components(H_pad3, psi_E6, slabs, coeffs)
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
        curlE3, psi_H6 = _curl_E_components(E_pad3, psi_H6, slabs, coeffs)
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

    # Initial state, mirroring arrays.reset(): zero fields, zero slab psi,
    # zeroed detector states. Building zeros directly (instead of reset())
    # avoids allocating full-volume (6,N) psi zeros that the slab layout
    # never uses.
    def _psi_init():
        return tuple(
            tuple(
                jnp.zeros(tuple(s.stop - s.start for s in sl), dtype=dtype)
                for sl in slabs[dax]
            )
            for dax in _PSI_DERIV_AXES
        )

    init = (
        tuple(jnp.zeros(vol_shape, dtype=dtype) for _ in range(3)),
        tuple(jnp.zeros(vol_shape, dtype=dtype) for _ in range(3)),
        _psi_init(),
        _psi_init(),
        {k: {k2: v2 * 0 for k2, v2 in v.items()} for k, v in arrays.detector_states.items()},
    )
    total_steps = config.time_steps_total
    E3, H3, psi_E6, psi_H6, det_states = jax.lax.fori_loop(0, total_steps, body, init)

    # Repack into a normal ArrayContainer so downstream detector-reading code
    # (and field inspection) sees the standard layout. Constructed directly —
    # not via aset — because in reclaim mode the input container holds
    # deleted leaves that aset's tree machinery would touch.
    if reclaim_memory:
        alpha, kappa, sigma = _rebuild_pml_arrays(vol_shape, dtype, objects)
    else:
        alpha, kappa, sigma = arrays.alpha, arrays.kappa, arrays.sigma
    out = ArrayContainer(
        fields=FieldState(
            E=jnp.stack(E3, axis=0),
            H=jnp.stack(H3, axis=0),
            psi_E=_scatter_psi(psi_E6, slabs, vol_shape, dtype),
            psi_H=_scatter_psi(psi_H6, slabs, vol_shape, dtype),
        ),
        alpha=alpha,
        kappa=kappa,
        sigma=sigma,
        inv_permittivities=inv_eps,
        inv_permeabilities=inv_mu,
        detector_states=det_states,
        recording_state=arrays.recording_state,
        # everything below is validated to be None/absent for the supported subset
        electric_conductivity=arrays.electric_conductivity,
        magnetic_conductivity=arrays.magnetic_conductivity,
        dispersive_P_curr=arrays.dispersive_P_curr,
        dispersive_P_prev=arrays.dispersive_P_prev,
        dispersive_c1=arrays.dispersive_c1,
        dispersive_c2=arrays.dispersive_c2,
        dispersive_c3=arrays.dispersive_c3,
        dispersive_inv_c2=arrays.dispersive_inv_c2,
    )
    return jnp.asarray(total_steps, dtype=jnp.int32), out
