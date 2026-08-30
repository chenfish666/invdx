"""The toy engine ported to JAX — same physics, differentiable.

Verified bit-equivalent to the numpy original (fdtd2d.py): max field
difference 9e-16 at float64 on a phc_bend gap measurement
(scripts/08_toy_jax_lesson1.py reproduces the check; tests/test_toy_jax.py
enforces it forever). The teaching version of this file — with the three
physics sections blanked as homework — lives in tutorials/01-jax-port/.

The only conceptual change vs numpy: JAX arrays are immutable.
    numpy:  Hx -= ...        Ez[1:-1,1:-1] += ...
    JAX:    Hx = Hx - ...    Ez = Ez.at[1:-1,1:-1].add(...)
The time loop is jax.lax.scan — the whole loop compiles into one XLA
program, which is what lets jax.grad differentiate through the entire
time evolution (the adjoint method for free; see scripts/09_toy_adjoint.py).
"""

import jax
import jax.numpy as jnp
import numpy as np


def gaussian_pulse(t, t0, spread, fcen=None):
    """與 numpy 版同義,但用 jnp(t 可以是整條時間軸向量)。"""
    env = jnp.exp(-(((t - t0) / spread) ** 2))
    if fcen is None:
        return env
    return jnp.sin(2 * jnp.pi * fcen * (t - t0)) * env


def simulate(nx, ny, dx, steps, source, probes=(), courant=0.5, eps=None,
             line_probes=None):
    """Pure-JAX core: eps in, per-step observables out — fully traceable,
    so jax.grad can differentiate any function of the outputs with respect
    to eps (the adjoint method, automatically). run() wraps this with the
    numpy packaging that matches the numpy engine's API; differentiable
    code (scripts/09) calls simulate() directly, because converting to
    numpy would cut the gradient chain.

    Returns (final_state, ys) where ys is a dict of stacked per-step
    outputs: "probes" (steps, n_probes), "energy" (steps,), and one
    (E_line, H_line) pair of (steps, n) arrays per line probe.
    """
    dt = courant * dx
    if eps is None:
        eps = jnp.ones((nx, ny))
    else:
        eps = jnp.asarray(eps)
    mur = (dt - dx) / (dt + dx)

    # 源的時間波形整條先算好,scan 每步吃一個值(比每步重算便宜也乾淨)
    amps = gaussian_pulse(jnp.arange(steps) * dt, source["t0"],
                          source["spread"], source.get("fcen"))

    line_probes = line_probes or {}
    probes = tuple(tuple(p) for p in probes)

    def inject(Ez, a):
        if "j0" in source:
            return Ez.at[source["i"], source["j0"]:source["j1"]].add(a)
        return Ez.at[source["i"], source["j"]].add(a)

    def step(state, a):
        """一個時間步:state 進、state 出(scan 的合約)。"""
        Ez, Hx, Hy = state

        # A:H 場更新(法拉第定律)——不可變風格,整陣列重算故不需 .at[]
        Hx = Hx - (dt / dx) * (Ez[:, 1:] - Ez[:, :-1])
        Hy = Hy + (dt / dx) * (Ez[1:, :] - Ez[:-1, :])

        Ez_old = Ez   # Mur 邊界需要「上一步的 Ez」——在內部更新前留影

        # B:E 場內部更新(安培定律)——材料唯一進場的位置就是 /eps
        curl = ((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                - (Hx[1:-1, 1:] - Hx[1:-1, :-1]))
        Ez = Ez.at[1:-1, 1:-1].add((dt / dx) / eps[1:-1, 1:-1] * curl)

        Ez = inject(Ez, a)

        # C:一階 Mur 吸收邊界(四條邊)
        Ez = Ez.at[0, :].set(Ez_old[1, :] + mur * (Ez[1, :] - Ez_old[0, :]))
        Ez = Ez.at[-1, :].set(Ez_old[-2, :]
                              + mur * (Ez[-2, :] - Ez_old[-1, :]))
        Ez = Ez.at[:, 0].set(Ez_old[:, 1] + mur * (Ez[:, 1] - Ez_old[:, 0]))
        Ez = Ez.at[:, -1].set(Ez_old[:, -2]
                              + mur * (Ez[:, -2] - Ez_old[:, -1]))

        # 每步的觀測值(scan 會自動把它們沿時間軸疊起來)
        out = {"probes": jnp.stack([Ez[p] for p in probes]) if probes
               else jnp.zeros((0,)),
               "energy": 0.5 * (jnp.sum(eps * Ez ** 2) + jnp.sum(Hx ** 2)
                                + jnp.sum(Hy ** 2)) * dx * dx}
        for name, (axis, k, lo, hi) in line_probes.items():
            if axis == "x":
                out[name] = (Ez[k, lo:hi], Hy[min(k, nx - 2), lo:hi])
            else:
                out[name] = (Ez[lo:hi, k], Hx[lo:hi, min(k, ny - 2)])
        return (Ez, Hx, Hy), out

    init = (jnp.zeros((nx, ny)), jnp.zeros((nx, ny - 1)),
            jnp.zeros((nx - 1, ny)))
    return jax.lax.scan(step, init, amps)


def run(nx, ny, dx, steps, source, probes=(), courant=0.5, eps=None,
        line_probes=None):
    """介面與 invdx.toy.fdtd2d.run 完全相同(換引擎不換 API)。"""
    if eps is not None:
        eps_np = np.asarray(eps, dtype=float)
        edge = np.concatenate([eps_np[0], eps_np[-1], eps_np[:, 0],
                               eps_np[:, -1]])
        if not np.allclose(edge, 1.0):
            raise ValueError("eps must be 1.0 on the outermost cells")

    probes = tuple(tuple(p) for p in probes)
    line_probes = line_probes or {}
    (Ez, _, _), ys = simulate(nx, ny, dx, steps, source, probes=probes,
                              courant=courant, eps=eps,
                              line_probes=line_probes)

    # 打包成與 numpy 版一模一樣的輸出(下游程式碼不用知道引擎換了)
    return {"Ez": np.asarray(Ez), "t": np.arange(steps) * (courant * dx),
            "probes": {p: np.asarray(ys["probes"][:, i])
                       for i, p in enumerate(probes)},
            "energy": np.asarray(ys["energy"]),
            "lines": {name: {"E": np.asarray(ys[name][0]),
                             "H": np.asarray(ys[name][1])}
                      for name in line_probes}}


def line_flux_spectrum_jnp(E_ts, H_ts, freqs, dt, dx, sign=1.0):
    """Differentiable twin of fdtd2d.line_flux_spectrum (jnp throughout)."""
    steps = E_ts.shape[0]
    t = jnp.arange(steps) * dt
    ker = jnp.exp(-2j * jnp.pi * jnp.asarray(freqs)[:, None] * t[None, :]) * dt
    Ef = ker @ E_ts.astype(ker.dtype)
    Hf = ker @ H_ts.astype(ker.dtype)
    return sign * 0.5 * jnp.real(jnp.sum(Ef * jnp.conj(Hf), axis=1)) * dx
