"""M-toy-2 第一課:把 toy 引擎移植到 JAX(你的作業本)。

這個檔案是**挖空的骨架**:鷹架(狀態初始化、lax.scan 迴圈、輸出打包)
已寫好,三個空格 A/B/C 是物理本體,由你來填——它們正對應 fdtd2d.py
(numpy 版)裡的同名段落。課程說明、概念、提示、檢查點都在
docs/toy-jax-lesson1.md;驗收指令:

    python scripts/08_toy_jax_lesson1.py

填完後它會用 phc_bend 的能隙量測驗證你的版本與 numpy 版逐位元一致
(float64 下差異應在 1e-15 量級)。

與 numpy 版唯一的「思想差異」:JAX 陣列不可變(functional style)。
    numpy:  Hx -= ...        Ez[1:-1,1:-1] += ...
    JAX:    Hx = Hx - ...    Ez = Ez.at[1:-1,1:-1].add(...)
時間迴圈交給 jax.lax.scan——它把整個迴圈編譯成一個 XLA 程式,
這正是之後 jax.grad 能「穿過整段時間演化」自動求梯度的前提。
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


def run(nx, ny, dx, steps, source, probes=(), courant=0.5, eps=None,
        line_probes=None):
    """介面與 invdx.toy.fdtd2d.run 完全相同(換引擎不換 API)。"""
    dt = courant * dx
    if eps is None:
        eps = jnp.ones((nx, ny))
    else:
        eps_np = np.asarray(eps, dtype=float)
        edge = np.concatenate([eps_np[0], eps_np[-1], eps_np[:, 0],
                               eps_np[:, -1]])
        if not np.allclose(edge, 1.0):
            raise ValueError("eps must be 1.0 on the outermost cells")
        eps = jnp.asarray(eps_np)
    mur = (dt - dx) / (dt + dx)

    # 源的時間波形整條先算好,scan 每步吃一個值(比每步重算便宜也乾淨)
    t_axis = np.arange(steps) * dt
    amps = gaussian_pulse(jnp.asarray(t_axis), source["t0"], source["spread"],
                          source.get("fcen"))

    line_probes = line_probes or {}
    probes = tuple(tuple(p) for p in probes)

    def inject(Ez, a):
        if "j0" in source:
            return Ez.at[source["i"], source["j0"]:source["j1"]].add(a)
        return Ez.at[source["i"], source["j"]].add(a)

    def step(state, a):
        """一個時間步:state 進、state 出(scan 的合約)。"""
        Ez, Hx, Hy = state

        # ================= 空格 A:H 場更新(法拉第定律)=================
        # 對照 fdtd2d.py 的兩行「H from curl E」。
        # 記住:JAX 不可變 → Hx = Hx - ...(不是 Hx -= ...)
        raise NotImplementedError("空格 A —— 見 docs/toy-jax-lesson1.md")
        # Hx = ...
        # Hy = ...
        # ================================================================

        Ez_old = Ez   # Mur 邊界需要「上一步的 Ez」——在內部更新前留影

        # ============ 空格 B:E 場內部更新(安培定律,材料在這)============
        # 對照 fdtd2d.py 的「E interior from curl H」。
        # 內部切片賦值用 Ez = Ez.at[1:-1, 1:-1].add(...)
        # 別忘了除以 eps[1:-1, 1:-1] —— 材料唯一進場的位置。
        raise NotImplementedError("空格 B —— 見 docs/toy-jax-lesson1.md")
        # curl = ...
        # Ez = ...
        # ================================================================

        Ez = inject(Ez, a)

        # ============ 空格 C:一階 Mur 吸收邊界(四條邊)============
        # 對照 fdtd2d.py 的四行 Mur。每條邊:
        #   Ez = Ez.at[邊].set(Ez_old[內鄰] + mur * (Ez[內鄰] - Ez_old[邊]))
        raise NotImplementedError("空格 C —— 見 docs/toy-jax-lesson1.md")
        # Ez = ...
        # Ez = ...
        # Ez = ...
        # Ez = ...
        # ================================================================

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
    (Ez, _, _), ys = jax.lax.scan(step, init, amps)

    # 打包成與 numpy 版一模一樣的輸出(下游程式碼不用知道引擎換了)
    return {"Ez": np.asarray(Ez), "t": t_axis,
            "probes": {p: np.asarray(ys["probes"][:, i])
                       for i, p in enumerate(probes)},
            "energy": np.asarray(ys["energy"]),
            "lines": {name: {"E": np.asarray(ys[name][0]),
                             "H": np.asarray(ys[name][1])}
                      for name in line_probes}}
