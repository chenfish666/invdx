> [English](RESULTS.md) · **繁體中文**

# 第二課的參考輸出

執行:`python scripts/09_toy_adjoint.py`(CPU/float64,全程 ~3 分鐘)
完整結果:指令第一行印出的那個 runs/ 目錄

## 實際輸出

```
[run] outputs -> runs/20260830-110925-toy-adjoint
[base] intact bend   mean T = 0.972
[base] damaged bend  mean T = 0.613   (defect (1, 0), benchmark Layer-I)
[adjoint] one backward pass = gradients for all 400 design-region parameters  (0.5s, incl. compile)
[gradcheck] pixel (np.int64(17), np.int64(12)): adjoint -3.668794e-07  FD -3.668793e-07  rel err 1.61e-07
[gradcheck] pixel (np.int64(10), np.int64(5)): adjoint +5.088147e-08  FD +5.088152e-08  rel err 9.79e-07
[gradcheck] pixel (np.int64(6), np.int64(0)): adjoint -9.336720e-07  FD -9.336726e-07  rel err 6.67e-07
[gradcheck] PASS (worst 9.79e-07 < 1e-5)
[opt] iter   0  mean T = 0.613
[opt] iter   5  mean T = 0.615
[opt] iter  10  mean T = 0.616
[opt] iter  15  mean T = 0.618
[opt] iter  20  mean T = 0.624
[opt] iter  25  mean T = 0.645
[opt] iter  30  mean T = 0.716
[opt] iter  35  mean T = 0.840
[opt] iter  40  mean T = 0.874
[opt] iter  45  mean T = 0.936
[opt] iter  50  mean T = 0.997
[opt] iter  55  mean T = 0.983
[opt] iter  59  mean T = 0.997

[result] damaged 0.613 -> healed 0.999 (intact 0.972)
[design] healed-region density (2x2 lattice, one entry = one a x a cell):
         0.00 0.03
         0.07 0.15
[done] runs/20260830-110925-toy-adjoint/results.json
```

跟自己跑出來的東西對照時,有兩個地方本來就會不一樣:run 目錄名(時間戳)、
還有 `[adjoint]` 那個秒數。其餘的數字在 CPU + float64 下是 deterministic 的
(同一台機器、同一版本重跑,逐位元相同),我們也用兩次獨立執行驗過。
另外,`[gradcheck]` 那三行裡的 `np.int64(...)` 是 numpy 純量的 repr,
不是索引的一部分——它們就是像素 (17, 12)、(10, 5)、(6, 0)。

加 `--gradcheck-only` 的話,輸出停在 `[gradcheck] PASS` 那行——
它寫完 `results.json` 就收工,不會印 `[done]`。

## 三個值得記住的點

1. **adjoint(伴隨法)的威力,量化**:設計區 400 個參數,梯度一次反向傳播
   0.5 秒全拿。同樣的資訊用有限差分要 800 次正向模擬——中央差分
   每個參數兩次。這個 800:1 的比值隨參數數目線性放大——上千維的
   三維設計就是上千比一。這就是「為什麼逆向設計非 adjoint 不可」的
   第一手體驗。

2. **梯度驗證先於一切**:三個隨機像素 adjoint vs 中央差分,
   相對誤差落在 1e-7 到 1e-6 之間。G2 閘門對 fdtdx 做的就是同一件事,
   方法一樣、判準較鬆(見
   [src/invdx/gates/g2_gradcheck.py](../../src/invdx/gates/g2_gradcheck.py)
   裡的 `REL_TOL = 0.05`),因為那邊是 GPU 上的 float32,而這裡是
   CPU 上的 float64。現在你的 toy 引擎也有資格當梯度的第三方證人。

3. **修復 ≠ 復原**:最優解沒有把缺陷柱長回來,反而把彎角附近
   **再清掉**一些材料(密度 0.00–0.15,四格幾乎全空),
   結果 0.999,**超過**完好晶格的 0.972。「移除柱子能改善 PhC 彎」
   是文獻裡的經典結論(Mekis et al., Phys. Rev. Lett. 77, 3787,
   1996),這裡是梯度自己重新發現的——而它根本不知道答案該長什麼樣。
   逆向設計的價值正在於此:它不受「應該長什麼樣」的成見約束。

## 誠實註記

- 這是 mean-T(三頻率平均)的單目標優化;沒有製程約束(最小線寬
  規則、二值化)——那些由 [src/invdx/fab/](../../src/invdx/fab/) 的
  濾波與投影工具負責。
- 密度 0~1 之間的灰色值在真實製程不存在;
  [src/invdx/fab/filters_jax.py](../../src/invdx/fab/filters_jax.py) 的
  β-投影(tanh projection)排程就是為了逼走灰色。toy 課程刻意先不加,
  讓 adjoint 本身站在聚光燈下。
- toy 引擎的一階 Mur 邊界有殘餘反射,所以 0.999 vs 0.972 這種細微差距
  落在引擎自己的系統誤差裡,不是可以拿去信的物理數字。誠實的讀法是
  「修復後的彎至少不比完好的差」,而不是一個精確的 +0.027。
