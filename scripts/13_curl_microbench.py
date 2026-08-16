"""Microbenchmark: fdtdx 0.6.2 curl_E + psi update + H update, stack vs stack-free.

Isolates one representative H-half-step of the fdtdx FDTD inner loop:
    curl, psi = curl_E(E_pad, psi, alpha, kappa, sigma)  # simulate_boundaries=True
    H = H + C * curl
V0 is a verbatim port of the installed
~/miniforge3/envs/invdx/lib/python3.12/site-packages/fdtdx/core/physics/curl.py

Variants:
  V0: verbatim (per-step b/a, jnp.stack for psi_updated and curl)
  V1: psi via chained .at[i].set writes; curl still jnp.stack-ed
  V2: psi via .at writes AND H updated per component (curl never materialized)
  V3: V2 + precomputed b, a, kappa_inv passed in (no per-step expm1/divisions)
  V4: V3 + component-tuple state: psi as tuple of six 3D arrays, H as tuple of 3
      (no (6,N)/(3,N) stacked state arrays at all)

Run: CUDA_VISIBLE_DEVICES=<idle gpu> ~/miniforge3/envs/invdx/bin/python /tmp/curl_bench.py
"""

import time

import numpy as np

import jax
import jax.numpy as jnp

try:
    from fdtdx.constants import c as C0
    from fdtdx.constants import eps0 as EPS0
except ImportError:  # standalone fallback, same values
    C0 = 299792458.0
    EPS0 = 8.8541878128e-12

COURANT = 0.99 / np.sqrt(3.0)  # fdtdx default courant_factor=0.99
RESOLUTION = 100e-9
# scalar prefactor of the exponent in b (constant-folded by XLA either way)
COEF = np.float32(-COURANT * RESOLUTION / C0 / EPS0)
CH = np.float32(0.5)  # representative H-update constant
ITERS = 50


# ---------------------------------------------------------------- shared pieces
def diffs_E(E_pad):
    """Verbatim finite differences from curl_E."""
    dyEz = (jnp.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, 1:-1]
    dzEy = (jnp.roll(E_pad[1], -1, axis=2) - E_pad[1])[1:-1, 1:-1, 1:-1]
    dzEx = (jnp.roll(E_pad[0], -1, axis=2) - E_pad[0])[1:-1, 1:-1, 1:-1]
    dxEz = (jnp.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, 1:-1]
    dxEy = (jnp.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, 1:-1]
    dyEx = (jnp.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, 1:-1]
    return dyEz, dzEy, dzEx, dxEz, dxEy, dyEx


def ba_coeffs(alpha, kappa, sigma):
    """Verbatim per-step PML coefficient computation from curl_E."""
    b = jnp.expm1(COEF * (sigma / kappa + alpha)) + 1
    a = jnp.nan_to_num(
        (b - 1.0) * sigma / (sigma + alpha * kappa) / kappa,
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    return b, a


def psi_components(psi_H, b, a, dyEz, dzEy, dzEx, dxEz, dxEy, dyEx):
    """Verbatim psi update math (component index mapping as installed)."""
    psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy = psi_H
    psi_Hxy = b[4] * psi_Hxy + a[4] * dyEz
    psi_Hxz = b[5] * psi_Hxz + a[5] * dzEy
    psi_Hyz = b[5] * psi_Hyz + a[5] * dzEx
    psi_Hyx = b[3] * psi_Hyx + a[3] * dxEz
    psi_Hzx = b[3] * psi_Hzx + a[3] * dxEy
    psi_Hzy = b[4] * psi_Hzy + a[4] * dyEx
    return psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy


# ------------------------------------------------------------------- variants
def v0_step(E_pad, H, psi_H, alpha, kappa, sigma):
    """Verbatim port of installed curl_E (simulate_boundaries=True) + H update."""
    dyEz, dzEy, dzEx, dxEz, dxEy, dyEx = diffs_E(E_pad)
    b, a = ba_coeffs(alpha, kappa, sigma)
    pxy, pxz, pyz, pyx, pzx, pzy = psi_components(
        psi_H, b, a, dyEz, dzEy, dzEx, dxEz, dxEy, dyEx)
    psi_new = jnp.stack((pxy, pxz, pyz, pyx, pzx, pzy), axis=0)
    curl_x = (1.0 / kappa[1] * dyEz + pxy) - (1.0 / kappa[2] * dzEy + pxz)
    curl_y = (1.0 / kappa[2] * dzEx + pyz) - (1.0 / kappa[0] * dxEz + pyx)
    curl_z = (1.0 / kappa[0] * dxEy + pzx) - (1.0 / kappa[1] * dyEx + pzy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)
    H_new = H + CH * curl
    return H_new, psi_new


def v1_step(E_pad, H, psi_H, alpha, kappa, sigma):
    """Identical math; psi_new via .at[i].set instead of jnp.stack. curl still stacked."""
    dyEz, dzEy, dzEx, dxEz, dxEy, dyEx = diffs_E(E_pad)
    b, a = ba_coeffs(alpha, kappa, sigma)
    pxy, pxz, pyz, pyx, pzx, pzy = psi_components(
        psi_H, b, a, dyEz, dzEy, dzEx, dxEz, dxEy, dyEx)
    psi_new = (psi_H.at[0].set(pxy).at[1].set(pxz).at[2].set(pyz)
                    .at[3].set(pyx).at[4].set(pzx).at[5].set(pzy))
    curl_x = (1.0 / kappa[1] * dyEz + pxy) - (1.0 / kappa[2] * dzEy + pxz)
    curl_y = (1.0 / kappa[2] * dzEx + pyz) - (1.0 / kappa[0] * dxEz + pyx)
    curl_z = (1.0 / kappa[0] * dxEy + pzx) - (1.0 / kappa[1] * dyEx + pzy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)
    H_new = H + CH * curl
    return H_new, psi_new


def v2_step(E_pad, H, psi_H, alpha, kappa, sigma):
    """Fully stack-free: psi via .at writes, H updated per component."""
    dyEz, dzEy, dzEx, dxEz, dxEy, dyEx = diffs_E(E_pad)
    b, a = ba_coeffs(alpha, kappa, sigma)
    pxy, pxz, pyz, pyx, pzx, pzy = psi_components(
        psi_H, b, a, dyEz, dzEy, dzEx, dxEz, dxEy, dyEx)
    psi_new = (psi_H.at[0].set(pxy).at[1].set(pxz).at[2].set(pyz)
                    .at[3].set(pyx).at[4].set(pzx).at[5].set(pzy))
    curl_x = (1.0 / kappa[1] * dyEz + pxy) - (1.0 / kappa[2] * dzEy + pxz)
    curl_y = (1.0 / kappa[2] * dzEx + pyz) - (1.0 / kappa[0] * dxEz + pyx)
    curl_z = (1.0 / kappa[0] * dxEy + pzx) - (1.0 / kappa[1] * dyEx + pzy)
    H_new = H.at[0].add(CH * curl_x).at[1].add(CH * curl_y).at[2].add(CH * curl_z)
    return H_new, psi_new


def precompute(alpha, kappa, sigma):
    b, a = ba_coeffs(alpha, kappa, sigma)
    kappa_inv = 1.0 / kappa
    return b, a, kappa_inv


def v3_step(E_pad, H, psi_H, b, a, kappa_inv):
    """V2 + precomputed b, a, kappa_inv (no per-step expm1/divisions)."""
    dyEz, dzEy, dzEx, dxEz, dxEy, dyEx = diffs_E(E_pad)
    pxy, pxz, pyz, pyx, pzx, pzy = psi_components(
        psi_H, b, a, dyEz, dzEy, dzEx, dxEz, dxEy, dyEx)
    psi_new = (psi_H.at[0].set(pxy).at[1].set(pxz).at[2].set(pyz)
                    .at[3].set(pyx).at[4].set(pzx).at[5].set(pzy))
    curl_x = (kappa_inv[1] * dyEz + pxy) - (kappa_inv[2] * dzEy + pxz)
    curl_y = (kappa_inv[2] * dzEx + pyz) - (kappa_inv[0] * dxEz + pyx)
    curl_z = (kappa_inv[0] * dxEy + pzx) - (kappa_inv[1] * dyEx + pzy)
    H_new = H.at[0].add(CH * curl_x).at[1].add(CH * curl_y).at[2].add(CH * curl_z)
    return H_new, psi_new


def v4_step(E_pad, H3, psi6, b6, a6, ki3):
    """V3 + component-tuple state: no (6,N)/(3,N) arrays across the jit boundary."""
    dyEz, dzEy, dzEx, dxEz, dxEy, dyEx = diffs_E(E_pad)
    Hx, Hy, Hz = H3
    pxy, pxz, pyz, pyx, pzx, pzy = psi6
    b3, b4, b5 = b6[3], b6[4], b6[5]
    a3, a4, a5 = a6[3], a6[4], a6[5]
    ki0, ki1, ki2 = ki3
    pxy = b4 * pxy + a4 * dyEz
    pxz = b5 * pxz + a5 * dzEy
    pyz = b5 * pyz + a5 * dzEx
    pyx = b3 * pyx + a3 * dxEz
    pzx = b3 * pzx + a3 * dxEy
    pzy = b4 * pzy + a4 * dyEx
    curl_x = (ki1 * dyEz + pxy) - (ki2 * dzEy + pxz)
    curl_y = (ki2 * dzEx + pyz) - (ki0 * dxEz + pyx)
    curl_z = (ki0 * dxEy + pzx) - (ki1 * dyEx + pzy)
    Hx = Hx + CH * curl_x
    Hy = Hy + CH * curl_y
    Hz = Hz + CH * curl_z
    return (Hx, Hy, Hz), (pxy, pxz, pyz, pyx, pzx, pzy)


# ------------------------------------------------------------------ harness
def bench(fn, args, iters=ITERS):
    out = fn(*args)  # warmup / compile
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        jax.block_until_ready(out)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def maxdiff(x, y):
    return float(jnp.max(jnp.abs(x - y)))


def make_inputs(n, seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    E_pad = jax.random.normal(ks[0], (3, n + 2, n + 2, n + 2), jnp.float32)
    H = jax.random.normal(ks[1], (3, n, n, n), jnp.float32)
    psi = jax.random.normal(ks[2], (6, n, n, n), jnp.float32) * 0.1
    # positive PML params, kappa >= 1: realistic decay factors, no NaN paths hit
    sigma = jax.random.uniform(ks[3], (6, n, n, n), jnp.float32, minval=0.0, maxval=1e4)
    alpha = jax.random.uniform(ks[4], (6, n, n, n), jnp.float32, minval=0.0, maxval=1e2)
    kappa = jax.random.uniform(ks[5], (6, n, n, n), jnp.float32, minval=1.0, maxval=3.0)
    return E_pad, H, psi, alpha, kappa, sigma


def run_size(n, which=("V0", "V1", "V2", "V3", "V4")):
    E_pad, H, psi, alpha, kappa, sigma = make_inputs(n)
    b, a, kappa_inv = jax.jit(precompute)(alpha, kappa, sigma)
    H3 = tuple(H[i] for i in range(3))
    psi6 = tuple(psi[i] for i in range(6))
    b6 = tuple(b[i] for i in range(6))
    a6 = tuple(a[i] for i in range(6))
    ki3 = tuple(kappa_inv[i] for i in range(3))
    jax.block_until_ready((H3, psi6, b6, a6, ki3))

    fns = {
        "V0": (jax.jit(v0_step), (E_pad, H, psi, alpha, kappa, sigma)),
        "V1": (jax.jit(v1_step), (E_pad, H, psi, alpha, kappa, sigma)),
        "V2": (jax.jit(v2_step), (E_pad, H, psi, alpha, kappa, sigma)),
        "V3": (jax.jit(v3_step), (E_pad, H, psi, b, a, kappa_inv)),
        "V4": (jax.jit(v4_step), (E_pad, H3, psi6, b6, a6, ki3)),
    }

    # reference output
    f0, args0 = fns["V0"]
    H_ref, psi_ref = f0(*args0)
    jax.block_until_ready((H_ref, psi_ref))

    ncell = n ** 3
    rows = []
    for name in which:
        fn, args = fns[name]
        H_out, psi_out = fn(*args)
        if name == "V4":
            H_out = jnp.stack(H_out, axis=0)
            psi_out = jnp.stack(psi_out, axis=0)
        dH = maxdiff(H_out, H_ref)
        dpsi = maxdiff(psi_out, psi_ref)
        del H_out, psi_out
        t = bench(fn, args)
        rows.append((name, t * 1e3, ncell / 1e6 / t, dH, dpsi))
    return rows


def run_size_donated(n, which=("V0", "V1", "V2", "V3", "V4")):
    """Same variants, but H/psi donated and fed forward call-to-call.

    This mimics the real fdtdx time loop, where H/psi live in a lax.scan /
    while_loop carry and XLA aliases the buffers in place. Without donation,
    every .at[] update pays a defensive full copy of the state array.
    """
    E_pad, H, psi, alpha, kappa, sigma = make_inputs(n)
    b, a, kappa_inv = jax.jit(precompute)(alpha, kappa, sigma)
    H3 = tuple(H[i] for i in range(3))
    psi6 = tuple(psi[i] for i in range(6))
    b6 = tuple(b[i] for i in range(6))
    a6 = tuple(a[i] for i in range(6))
    ki3 = tuple(kappa_inv[i] for i in range(3))
    jax.block_until_ready((H3, psi6, b6, a6, ki3))

    fns = {
        "V0": (jax.jit(v0_step, donate_argnums=(1, 2)), (alpha, kappa, sigma)),
        "V1": (jax.jit(v1_step, donate_argnums=(1, 2)), (alpha, kappa, sigma)),
        "V2": (jax.jit(v2_step, donate_argnums=(1, 2)), (alpha, kappa, sigma)),
        "V3": (jax.jit(v3_step, donate_argnums=(1, 2)), (b, a, kappa_inv)),
        "V4": (jax.jit(v4_step, donate_argnums=(1, 2)), (b6, a6, ki3)),
    }

    H_ref, psi_ref = jax.jit(v0_step)(E_pad, H, psi, alpha, kappa, sigma)
    jax.block_until_ready((H_ref, psi_ref))

    ncell = n ** 3
    rows = []
    for name in which:
        fn, extra = fns[name]
        if name == "V4":
            Hs = tuple(jnp.copy(x) for x in H3)
            Ps = tuple(jnp.copy(x) for x in psi6)
        else:
            Hs, Ps = jnp.copy(H), jnp.copy(psi)
        # warmup call doubles as the equivalence check (single step from t=0)
        Hs, Ps = fn(E_pad, Hs, Ps, *extra)
        jax.block_until_ready((Hs, Ps))
        if name == "V4":
            dH = maxdiff(jnp.stack(Hs, axis=0), H_ref)
            dpsi = maxdiff(jnp.stack(Ps, axis=0), psi_ref)
        else:
            dH = maxdiff(Hs, H_ref)
            dpsi = maxdiff(Ps, psi_ref)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            Hs, Ps = fn(E_pad, Hs, Ps, *extra)
            jax.block_until_ready((Hs, Ps))
        t1 = time.perf_counter()
        t = (t1 - t0) / ITERS
        rows.append((name, t * 1e3, ncell / 1e6 / t, dH, dpsi))
    return rows


def print_table(title, rows):
    t0 = rows[0][1]  # V0 must be first row
    print(f"\n=== {title} ===")
    print(f"{'variant':<8}{'ms/step':>10}{'Mcell/s':>10}{'vs V0':>8}{'max|dH|':>12}{'max|dpsi|':>12}")
    for name, ms, mcs, dH, dpsi in rows:
        print(f"{name:<8}{ms:>10.3f}{mcs:>10.1f}{t0 / ms:>7.2f}x{dH:>12.3e}{dpsi:>12.3e}")


# ---------------------------------------------------------------- V5: PML slabs
# CPML auxiliaries are mathematically nonzero only inside the PML slabs
# (verified on placed fdtdx arrays: base kappa=1/sigma=0/alpha=0 gives b=1,
# a=0 outside every slab, so psi starting at 0 stays exactly 0 there). V5
# stores psi and the b/a/1-kappa coefficients as slab-sized arrays only and
# computes the interior curl as the plain difference. Exactness in the slab
# corners comes from factorizing the curl per derivative axis: each factor
# T = ki*d + psi only involves ONE axis's coefficients, so per-axis slab
# .at[slice].set updates compose exactly.


def slab_slices(n, npml):
    """(lo, hi) full-3D slice tuples for each axis of an n^3 volume."""
    out = []
    for d in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[d] = slice(0, npml)
        hi[d] = slice(n - npml, n)
        out.append((tuple(lo), tuple(hi)))
    return out


def make_slab_inputs(n, npml, seed=0):
    """Slab-structured PML inputs: alpha/kappa/sigma at vacuum values (0/1/0)
    outside the six PML slabs, graded 1-D profiles broadcast along each face
    inside (matching fdtdx's modify_arrays, whose profiles depend on PML
    depth only); psi random inside slabs, 0 outside."""
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    E_pad = jax.random.normal(ks[0], (3, n + 2, n + 2, n + 2), jnp.float32)
    H = jax.random.normal(ks[1], (3, n, n, n), jnp.float32)
    slabs = slab_slices(n, npml)

    sigma = jnp.zeros((6, n, n, n), jnp.float32)
    alpha = jnp.zeros((6, n, n, n), jnp.float32)
    kappa = jnp.ones((6, n, n, n), jnp.float32)
    psi = jnp.zeros((6, n, n, n), jnp.float32)
    deriv_axes = (1, 2, 2, 0, 0, 1)  # psi_(xy, xz, yz, yx, zx, zy)
    k_iter = iter(jax.random.split(jax.random.PRNGKey(seed + 1), 200))

    def prof(key, lo, hi, d):
        """random 1-D depth profile, broadcast transversally over the slab."""
        shape1 = [1, 1, 1]
        shape1[d] = npml
        p = jax.random.uniform(key, tuple(shape1), jnp.float32, lo, hi)
        full_shape = tuple(npml if i == d else n for i in range(3))
        return jnp.broadcast_to(p, full_shape)

    for d in range(3):
        for sl in slabs[d]:
            for row in (d, d + 3):
                sigma = sigma.at[(row, *sl)].set(prof(next(k_iter), 0.0, 1e4, d))
                alpha = alpha.at[(row, *sl)].set(prof(next(k_iter), 0.0, 1e2, d))
                kappa = kappa.at[(row, *sl)].set(prof(next(k_iter), 1.0, 3.0, d))
    for i, dax in enumerate(deriv_axes):
        for sl in slabs[dax]:
            shape = tuple(npml if k == dax else n for k in range(3))
            psi = psi.at[(i, *sl)].set(
                jax.random.normal(next(k_iter), shape, jnp.float32) * 0.1)
    return E_pad, H, psi, alpha, kappa, sigma, slabs


def extract_slab_state(psi, b, a, kappa_inv, slabs):
    """Slab-sized psi + per-axis coefficient tuples for v5_step.

    psi components use H-side rows (3..5) of b/a; the curl's 1/kappa factor
    uses rows 0..2 (upstream uses kappa[0..2] in both curls).
    """
    deriv_axes = (1, 2, 2, 0, 0, 1)
    psi_slabs = tuple(
        tuple(psi[i][sl] for sl in slabs[dax]) for i, dax in enumerate(deriv_axes)
    )
    coeffs = tuple(
        tuple(
            (b[d + 3][sl], a[d + 3][sl], kappa_inv[d][sl]) for sl in slabs[d]
        )
        for d in range(3)
    )
    return psi_slabs, coeffs


def _slab_factor(diff, slabs_d, psi_pair, coeffs_d):
    """T = diff outside slabs; ki*diff + psi_new inside. Returns (T, psi_new)."""
    new_psi = []
    T = diff
    for sl, psi_s, (b_s, a_s, ki_s) in zip(slabs_d, psi_pair, coeffs_d):
        d_s = diff[sl]
        p_new = b_s * psi_s + a_s * d_s
        T = T.at[sl].set(ki_s * d_s + p_new)
        new_psi.append(p_new)
    return T, tuple(new_psi)


def make_v5_step(slabs):
    """v5: V4 + psi/coefficients restricted to PML slabs (slab-sized state)."""

    def v5_step(E_pad, H3, psi_slabs, coeffs):
        dyEz, dzEy, dzEx, dxEz, dxEy, dyEx = diffs_E(E_pad)
        Hx, Hy, Hz = H3
        p_xy, p_xz, p_yz, p_yx, p_zx, p_zy = psi_slabs
        cx, cy, cz = coeffs
        T1x, p_xy = _slab_factor(dyEz, slabs[1], p_xy, cy)
        T2x, p_xz = _slab_factor(dzEy, slabs[2], p_xz, cz)
        T1y, p_yz = _slab_factor(dzEx, slabs[2], p_yz, cz)
        T2y, p_yx = _slab_factor(dxEz, slabs[0], p_yx, cx)
        T1z, p_zx = _slab_factor(dxEy, slabs[0], p_zx, cx)
        T2z, p_zy = _slab_factor(dyEx, slabs[1], p_zy, cy)
        Hx = Hx + CH * (T1x - T2x)
        Hy = Hy + CH * (T1y - T2y)
        Hz = Hz + CH * (T1z - T2z)
        return (Hx, Hy, Hz), (p_xy, p_xz, p_yz, p_yx, p_zx, p_zy)

    return v5_step


def extract_slab_state_1d(psi, b, a, kappa_inv, slabs, n, npml):
    """Like extract_slab_state but coefficients as 1-D depth profiles kept in
    broadcastable (…,1,1) shape — CPML coefficients depend on PML depth only
    (gprMax, flaport/fdtd, Schneider ch.11 all store 1-D vectors)."""
    deriv_axes = (1, 2, 2, 0, 0, 1)
    psi_slabs = tuple(
        tuple(psi[i][sl] for sl in slabs[dax]) for i, dax in enumerate(deriv_axes)
    )

    def to_1d(block, d):
        idx = [slice(0, 1)] * 3
        idx[d] = slice(None)
        return block[tuple(idx)]  # shape e.g. (npml, 1, 1)

    coeffs = tuple(
        tuple(
            (to_1d(b[d + 3][sl], d), to_1d(a[d + 3][sl], d), to_1d(kappa_inv[d][sl], d))
            for sl in slabs[d]
        )
        for d in range(3)
    )
    return psi_slabs, coeffs


def scatter_psi(psi_slabs, slabs, n):
    """Slab psi state -> full (6, n, n, n) array (zeros outside)."""
    deriv_axes = (1, 2, 2, 0, 0, 1)
    full = jnp.zeros((6, n, n, n), jnp.float32)
    for i, dax in enumerate(deriv_axes):
        for sl, p in zip(slabs[dax], psi_slabs[i]):
            full = full.at[(i, *sl)].set(p)
    return full


def run_size_slab(n, npml, iters=ITERS):
    """V0 vs V4 vs V5 on slab-structured PML inputs, donated carry."""
    E_pad, H, psi, alpha, kappa, sigma, slabs = make_slab_inputs(n, npml)
    b, a, kappa_inv = jax.jit(precompute)(alpha, kappa, sigma)
    H3 = tuple(H[i] for i in range(3))
    psi6 = tuple(psi[i] for i in range(6))
    b6 = tuple(b[i] for i in range(6))
    a6 = tuple(a[i] for i in range(6))
    ki3 = tuple(kappa_inv[i] for i in range(3))
    psi_slabs, coeffs = extract_slab_state(psi, b, a, kappa_inv, slabs)
    _, coeffs1d = extract_slab_state_1d(psi, b, a, kappa_inv, slabs, n, npml)
    jax.block_until_ready((H3, psi6, b6, a6, ki3, psi_slabs, coeffs, coeffs1d))

    H_ref, psi_ref = jax.jit(v0_step)(E_pad, H, psi, alpha, kappa, sigma)
    jax.block_until_ready((H_ref, psi_ref))

    v5 = make_v5_step(slabs)
    fns = {
        "V0": (jax.jit(v0_step, donate_argnums=(1, 2)), (alpha, kappa, sigma)),
        "V4": (jax.jit(v4_step, donate_argnums=(1, 2)), (b6, a6, ki3)),
        "V5": (jax.jit(v5, donate_argnums=(1, 2)), (coeffs,)),
        "V6": (jax.jit(v5, donate_argnums=(1, 2)), (coeffs1d,)),
    }

    ncell = n**3
    rows = []
    for name, (fn, extra) in fns.items():
        if name in ("V5", "V6"):
            Hs = tuple(jnp.copy(x) for x in H3)
            Ps = jax.tree.map(jnp.copy, psi_slabs)
        elif name == "V4":
            Hs = tuple(jnp.copy(x) for x in H3)
            Ps = tuple(jnp.copy(x) for x in psi6)
        else:
            Hs, Ps = jnp.copy(H), jnp.copy(psi)
        Hs, Ps = fn(E_pad, Hs, Ps, *extra)
        jax.block_until_ready((Hs, Ps))
        if name == "V0":
            dH = maxdiff(Hs, H_ref)
            dpsi = maxdiff(Ps, psi_ref)
        elif name == "V4":
            dH = maxdiff(jnp.stack(Hs, axis=0), H_ref)
            dpsi = maxdiff(jnp.stack(Ps, axis=0), psi_ref)
        else:  # V5 / V6: slab psi state
            dH = maxdiff(jnp.stack(Hs, axis=0), H_ref)
            dpsi = maxdiff(scatter_psi(Ps, slabs, n), psi_ref)
        t0 = time.perf_counter()
        for _ in range(iters):
            Hs, Ps = fn(E_pad, Hs, Ps, *extra)
            jax.block_until_ready((Hs, Ps))
        t1 = time.perf_counter()
        t = (t1 - t0) / iters
        rows.append((name, t * 1e3, ncell / 1e6 / t, dH, dpsi))
    return rows


def main():
    dev = jax.devices()[0]
    print(f"jax {jax.__version__}  device: {dev.device_kind}  iters={ITERS}")

    rows256 = run_size(256)
    print_table("no donation, fixed inputs: (3, 256, 256, 256), 16.78 Mcells", rows256)

    rows256d = run_size_donated(256)
    print_table("donated carry (scan-like): (3, 256, 256, 256), 16.78 Mcells", rows256d)

    best = min(rows256d[1:], key=lambda r: r[1])[0]
    print(f"\nbest variant at 256^3 (donated): {best}")

    rows128 = run_size(128, which=("V0", best))
    print_table(f"no donation: (3, 128, 128, 128), 2.10 Mcells (V0 vs {best})", rows128)

    rows128d = run_size_donated(128, which=("V0", best))
    print_table(f"donated carry: (3, 128, 128, 128), 2.10 Mcells (V0 vs {best})", rows128d)


def main_slab():
    dev = jax.devices()[0]
    print(f"jax {jax.__version__}  device: {dev.device_kind}  iters={ITERS}")
    for n, npml in ((256, 16), (256, 25), (200, 25)):
        rows = run_size_slab(n, npml)
        print_table(
            f"slab-PML donated carry: {n}^3, npml={npml} "
            f"(slab fraction/axis {2 * npml / n:.0%})", rows)


if __name__ == "__main__":
    import sys

    if "--slab" in sys.argv:
        main_slab()
    else:
        main()
