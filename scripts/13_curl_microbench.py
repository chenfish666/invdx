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


if __name__ == "__main__":
    main()
