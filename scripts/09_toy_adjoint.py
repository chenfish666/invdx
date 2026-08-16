#!/usr/bin/env python
"""第二課主線:第一個伴隨梯度 —— 在教授的光子晶體 90° 彎上修復缺陷。

劇本:
  1. 拿 phc_bend 的彎,打掉論文裡傷害最大的 Layer-I 缺陷柱(透射重傷)
  2. 在缺陷周圍畫一個 2a x 2a 的設計區,材料變成連續參數
     eps = 1 + (eps_rod - 1) * sigmoid(theta)
  3. jax.grad 穿過整段 FDTD 時間演化,一次拿到設計區每個格點的梯度
     (這就是伴隨法:一次正向 + 一次反向 = 全部參數的梯度)
  4. 先用有限差分驗證梯度(invdx 的 G2 精神:梯度不驗證不上桌)
  5. Adam 迭代,看逆向設計自動把彎「治好」

  python scripts/09_toy_adjoint.py                  # 全流程(~3 分鐘,CPU)
  python scripts/09_toy_adjoint.py --gradcheck-only # 只做梯度驗證(~1 分鐘)
"""

import argparse
import os
import sys
import time

# 決定性優先:課程與梯度驗證都在 CPU + float64(GPU 是之後規模化的事)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import phc_bend
from invdx.toy import fdtd2d, fdtd2d_jax
from invdx import runio

DEFECT = (1, 0)                     # 論文 Layer-I 水平缺陷(傷害最大)
FSTARS = (0.31, 0.34, 0.37)         # 這個尺寸的能隙內三個頻率


def build_case(cfg):
    """量測配置 + 直波導歸一化(numpy 跑一次,當常數)。"""
    ports = phc_bend._toy_ports(cfg)
    fcen = 0.5 * (cfg.f_min + cfg.f_max)
    spread = 1.0 / (np.pi * (cfg.f_max - cfg.f_min) / 2)
    n = int(round((cfg.n_side + 2 * cfg.pad_a) * cfg.res_per_a))
    dx = 1.0 / cfg.res_per_a
    kw = dict(nx=n, ny=n, dx=dx, steps=cfg.toy_steps,
              source={**ports["src"], "t0": 4 * spread, "spread": spread,
                      "fcen": fcen},
              courant=cfg.toy_courant)
    ref = fdtd2d.run(**kw, eps=phc_bend.epsilon_grid(cfg, "straight"),
                     line_probes={"out": ports["straight_out"]})
    dt = cfg.toy_courant * dx
    p_ref = fdtd2d.line_flux_spectrum(ref["lines"]["out"], FSTARS, dt, dx,
                                      sign=-1.0)
    return kw, ports["bend_out"], dt, dx, p_ref


def design_box(cfg):
    """缺陷周圍 2a x 2a 的格點切片(晶格座標 ix in {c+1,c+2}, iy in {c-1,c})。"""
    res, c, pad = cfg.res_per_a, cfg.center, cfg.pad_a
    gx = slice(int((pad + c + 1) * res), int((pad + c + 3) * res))
    gy = slice(int((pad + c - 1) * res), int((pad + c + 1) * res))
    return gx, gy


def main():
    p = base_parser(__doc__)
    p.add_argument("--gradcheck-only", action="store_true")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.2)
    args = p.parse_args()
    cfg = apply_overrides(
        phc_bend.PhCBendConfig(n_side=9, res_per_a=10, toy_steps=2500),
        args)
    d = start_run(cfg, args, "toy-adjoint")

    kw, bend_out, dt, dx, p_ref = build_case(cfg)
    eps_damaged = phc_bend.epsilon_grid(cfg, "bend", defect=DEFECT)
    eps_intact = phc_bend.epsilon_grid(cfg, "bend")
    gx, gy = design_box(cfg)
    p_ref_j = jnp.asarray(p_ref)
    base = jnp.asarray(eps_damaged)

    def eps_of(theta):
        box = 1.0 + (cfg.eps_rod - 1.0) * jax.nn.sigmoid(theta)
        return base.at[gx, gy].set(box)

    def mean_T(eps):
        _, ys = fdtd2d_jax.simulate(**kw, eps=eps,
                                    line_probes={"out": bend_out})
        E, H = ys["out"]
        P = fdtd2d_jax.line_flux_spectrum_jnp(E, H, FSTARS, dt, dx,
                                              sign=+1.0)
        return jnp.mean(P / p_ref_j)

    objective = jax.jit(lambda th: mean_T(eps_of(th)))
    vg = jax.jit(jax.value_and_grad(lambda th: mean_T(eps_of(th))))

    # 起點 = 受損結構本身(theta 由受損 eps 反推)
    p0 = (np.asarray(eps_damaged[gx, gy]) - 1.0) / (cfg.eps_rod - 1.0)
    p0 = np.clip(p0, 1e-3, 1 - 1e-3)
    theta = jnp.asarray(np.log(p0 / (1 - p0)))

    T_damaged = float(objective(theta))
    T_intact = float(mean_T(jnp.asarray(eps_intact)))
    print(f"[base] 完好的彎  mean T = {T_intact:.3f}")
    print(f"[base] 受損的彎  mean T = {T_damaged:.3f}   "
          f"(缺陷 {DEFECT},論文 Layer-I)")

    # ---- 梯度驗證(G2 精神:不驗證的梯度不上桌)----
    t0 = time.time()
    val, g = vg(theta)
    t_grad = time.time() - t0
    print(f"[adjoint] 一次反向傳播 = 設計區全部 {g.size} 個參數的梯度  "
          f"({t_grad:.1f}s,含編譯)")
    rng = np.random.default_rng(0)
    idx = [tuple(rng.integers(0, s) for s in g.shape) for _ in range(3)]
    h = 1e-4
    worst = 0.0
    for ij in idx:
        e = jnp.zeros_like(theta).at[ij].set(h)
        fd = (float(objective(theta + e)) - float(objective(theta - e))) \
            / (2 * h)
        ad = float(g[ij])
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)
        worst = max(worst, rel)
        print(f"[gradcheck] pixel {ij}: adjoint {ad:+.6e}  "
              f"FD {fd:+.6e}  rel err {rel:.2e}")
    if worst > 1e-5:
        print("[gradcheck] FAIL — 梯度不可信,停止")
        return 1
    print(f"[gradcheck] PASS(最差 {worst:.2e} < 1e-5)")
    if args.gradcheck_only:
        runio.save_json(os.path.join(d, "results.json"),
                        {"gradcheck_worst_rel": worst})
        return 0

    # ---- 逆向設計:讓梯度自己把彎治好 ----
    import optax

    opt = optax.adam(args.lr)
    state = opt.init(theta)
    hist = []
    for it in range(args.iters):
        val, g = vg(theta)
        upd, state = opt.update(-g, state)     # 最大化 → 負梯度給 minimizer
        theta = optax.apply_updates(theta, upd)
        hist.append(float(val))
        if it % 5 == 0 or it == args.iters - 1:
            print(f"[opt] iter {it:3d}  mean T = {float(val):.3f}")

    T_healed = float(objective(theta))
    print(f"\n[result] 受損 {T_damaged:.3f} → 修復後 {T_healed:.3f} "
          f"(完好 {T_intact:.3f})")

    # 設計區長什麼樣(每個晶格 cell 的平均密度,0=真空 9=滿柱)
    dens = np.asarray(jax.nn.sigmoid(theta))
    r = cfg.res_per_a
    print("[design] 修復區密度(2x2 晶格,每格 = 一個 a x a cell):")
    for jy in range(1, -1, -1):
        row = " ".join(
            f"{dens[ix * r:(ix + 1) * r, jy * r:(jy + 1) * r].mean():.2f}"
            for ix in range(2))
        print("         " + row)

    runio.save_json(os.path.join(d, "results.json"), {
        "T_intact": T_intact, "T_damaged": T_damaged, "T_healed": T_healed,
        "gradcheck_worst_rel": worst, "history": hist,
        "fstars": list(FSTARS), "defect": list(DEFECT)})
    np.savez(os.path.join(d, "design.npz"), theta=np.asarray(theta),
             eps=np.asarray(eps_of(theta)))
    print(f"[done] {d}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
