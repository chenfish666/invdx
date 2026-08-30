> [English](RESULTS.md) · **繁體中文**

# 第一課的參考輸出

空格 A/B/C 的參考解在 [src/invdx/toy/fdtd2d_jax.py](../../src/invdx/toy/fdtd2d_jax.py);
本檔記錄一次驗收的實際輸出,親手練習時可以對照。

## 驗收輸出(scripts/08_toy_jax_lesson1.py --gpu)

```
[case] photonic-crystal bulk gap measurement, 110^2 grid, 2000 steps (small, runs in seconds)
[diff] max|dE| = 8.882e-16, max|dH| = 8.882e-16 (field scale 4.140e-01)
[time] numpy 0.38s | jax first run 1.99s (with compile) | jax rerun 0.34s

[PASS] both engines agree to the last bit -- your first JAX FDTD works.
       Next lesson (scripts/09): make eps a parameter, push jax.grad
       through the whole time evolution, then check it against finite differences.
[gpu] jax default device: gpu (<你的顯示卡型號>)
[gpu] same code, zero edits, on gpu: first run 0.33s, rerun 0.37s
[gpu] that is the port's second payoff: the numpy version is CPU-only, forever.
```

`<你的顯示卡型號>` 是佔位符:jax 印的是它找到那張卡的 `device_kind`
(JAX 裝置物件上的欄位,內容就是卡的型號字串);這份輸出是在一台會印出
`Quadro RTX 6000` 的機器上抄的。型號前面那個 `gpu` 是 `device.platform`,
固定小寫,不會是 `GPU` 也不會是 `cuda`。

## 數字怎麼讀

- **8.9e-16**:float64 的機器精度(≈2.2e-16)乘上幾步的累積。這不是
  「兩個引擎很接近」,是「兩個引擎是同一個物理」——每一步浮點運算
  都相同,只是執行方式不同(直譯 numpy vs XLA 編譯)。
- **jax 首跑 1.99s vs 再跑 0.34s**:差額 ≈1.6s 是 tracing(JAX 先把你的
  Python 走一遍、記錄成計算圖)加編譯,只付一次。同形狀的後續呼叫
  直接用編譯產物。
- **GPU 0.37s 沒有比 CPU 0.34s 快**:這個問題太小(110²),塞不滿
  GPU;搬資料的開銷吃掉了算力優勢。GPU 的回報要在大網格(全尺寸的
  三維問題動輒數千萬格點)或 float32 時才展現——「小問題別上 GPU」
  本身就是一課。絕對秒數會隨機器不同,值得看的是比值。
- **這條測試從此常駐**:tests/test_toy_jax.py 從此在每次 `make gates` 盯著
  兩引擎等價,JAX 版再也不能悄悄偏離 numpy 版。

## 這一課埋的伏筆

E 場更新裡那個 `/ eps[1:-1, 1:-1]` 現在活在一個 XLA 可微分程式裡。
第二課(`tutorials/02-first-adjoint`)就對它下手:`jax.grad(透射率)(eps)` 一次拿到
「每個格點的材料怎麼影響輸出」——這就是 adjoint(伴隨法),而且是自動的。
