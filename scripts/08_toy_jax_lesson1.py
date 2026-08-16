#!/usr/bin/env python
"""第一課驗收器:你的 JAX 引擎 vs numpy 引擎,同一個光子晶體能隙量測。

  python scripts/08_toy_jax_lesson1.py                # 驗收主線引擎
  python scripts/08_toy_jax_lesson1.py --scan-demo    # lax.scan 三行入門範例
  python scripts/08_toy_jax_lesson1.py --gpu          # 驗收 + 在 GPU 上跑同一份程式碼
  python scripts/08_toy_jax_lesson1.py \
      --file tutorials/01-jax-port/fdtd2d_jax_skeleton.py
                                  # 驗收你自己填的教學版骨架(不動主線)

通過標準:float64 下逐點差 < 1e-9(實際應在 1e-15 機器精度量級)。
"""

import argparse
import os
import sys
import time

# 課程預設在 CPU 上驗收(結果決定性、不用排隊等 GPU);--gpu 才開 GPU。
# 平台選擇必須在 import jax 之前生效,否則 CUDA plugin 會先探測並噴警告。
if "--gpu" not in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# float64 開關必須在任何 jax 陣列誕生之前設好 —— 這是 JAX 最經典的陷阱:
# 它預設 float32(GPU 友善),而 numpy 是 float64;不開這個,你的引擎
# 「看起來對」但和 numpy 版差在 1e-7,你會分不清是物理錯還是精度差。
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


def scan_demo():
    """lax.scan 是「帶狀態的迴圈」:cumsum 三行版。"""
    import jax.numpy as jnp

    def step(carry, x):          # carry 進、carry 出,外加一個每步輸出
        carry = carry + x
        return carry, carry      # (新狀態, 這一步想記錄的東西)

    total, history = jax.lax.scan(step, 0.0, jnp.arange(5.0))
    print("total   =", total)          # 10.0
    print("history =", history)        # [0 1 3 6 10]
    print()
    print("FDTD 的對應:carry = (Ez, Hx, Hy) 場狀態;x = 每步的源振幅;")
    print("每步輸出 = 探針讀值。scan 把整個時間迴圈編成一個 XLA 程式,")
    print("之後 jax.grad 就能穿過它對任何輸入(比如 eps)自動求梯度。")


def build_case():
    from invdx.problems import phc_bend

    cfg = phc_bend.PhCBendConfig(n_side=9, res_per_a=10, toy_steps=2000,
                                 n_freq=9)
    eps = phc_bend.epsilon_grid(cfg, "bulk")
    ports = phc_bend._toy_ports(cfg)
    fcen = 0.5 * (cfg.f_min + cfg.f_max)
    spread = 1.0 / (np.pi * (cfg.f_max - cfg.f_min) / 2)
    n = eps.shape[0]
    return dict(nx=n, ny=n, dx=1.0 / cfg.res_per_a, steps=cfg.toy_steps,
                source={**ports["bulk_src"], "t0": 4 * spread,
                        "spread": spread, "fcen": fcen},
                eps=eps, line_probes={"out": ports["bulk_out"]})


def load_engine(path):
    """Load a student-filled skeleton from a file path (tutorial mode)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("student_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(engine_file=None):
    from invdx.toy import fdtd2d

    if engine_file:
        fdtd2d_jax = load_engine(engine_file)
        print(f"[mode] 驗收教學版骨架:{engine_file}")
    else:
        from invdx.toy import fdtd2d_jax

    kw = build_case()
    print(f"[case] 光子晶體 bulk 能隙量測,格點 {kw['nx']}^2、"
          f"{kw['steps']} 步(小尺寸,秒級)")

    t0 = time.time()
    ref = fdtd2d.run(**kw)
    t_np = time.time() - t0

    try:
        t0 = time.time()
        mine = fdtd2d_jax.run(**kw)
        t_j1 = time.time() - t0
    except NotImplementedError as e:
        print(f"\n[todo] {e}")
        print("[todo] 三個空格還沒填完 —— 開 src/invdx/toy/fdtd2d_jax.py,"
              "照 docs/toy-jax-lesson1.md 一格一格來。")
        return 1

    t0 = time.time()
    mine = fdtd2d_jax.run(**kw)
    t_j2 = time.time() - t0

    E0, H0 = ref["lines"]["out"]["E"], ref["lines"]["out"]["H"]
    E1, H1 = mine["lines"]["out"]["E"], mine["lines"]["out"]["H"]
    scale = float(np.max(np.abs(E0)))
    dE = float(np.max(np.abs(E0 - E1)))
    dH = float(np.max(np.abs(H0 - H1)))
    print(f"[diff] max|dE| = {dE:.3e}, max|dH| = {dH:.3e} (場量級 {scale:.3e})")
    print(f"[time] numpy {t_np:.2f}s | jax 首跑 {t_j1:.2f}s(含編譯)| "
          f"jax 再跑 {t_j2:.2f}s")

    if dE < 1e-9 * scale and dH < 1e-9 * scale:
        print("\n[PASS] 兩個引擎逐位元同物理 —— 你的第一個 JAX FDTD 成立。")
        print("       下一課(M-toy-4 前哨):把 eps 當參數,jax.grad 穿過")
        print("       整段時間演化拿梯度,再用有限差分驗證它。")
        return 0
    print("\n[FAIL] 有差異 —— 檢查:(1) 空格順序(H 先、E 後、源、Mur)")
    print("       (2) Ez_old 是否在 E 內部更新『之前』留影")
    print("       (3) eps 是否除在 E 更新裡(不是 H)")
    return 1


def gpu_check():
    from invdx.toy import fdtd2d_jax

    dev = jax.devices()[0]
    print(f"[gpu] jax 預設裝置:{dev.platform} ({dev.device_kind})")
    kw = build_case()
    t0 = time.time()
    fdtd2d_jax.run(**kw)
    t1 = time.time() - t0
    t0 = time.time()
    fdtd2d_jax.run(**kw)
    t2 = time.time() - t0
    print(f"[gpu] 同一份程式碼零修改跑在 {dev.platform}:首跑 {t1:.2f}s、"
          f"再跑 {t2:.2f}s")
    print("[gpu] 這就是移植的第二個回報:numpy 版永遠只有 CPU。")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-demo", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--file", default=None,
                   help="path to a student-filled skeleton to check instead "
                        "of the mainline engine")
    args = p.parse_args()
    if args.scan_demo:
        scan_demo()
        return 0
    rc = check(args.file)
    if rc == 0 and args.gpu:
        gpu_check()
    return rc


if __name__ == "__main__":
    sys.exit(main())
