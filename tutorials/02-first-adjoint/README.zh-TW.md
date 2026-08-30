> [English](README.md) · **繁體中文**

# 第二課:第一個 adjoint 梯度 —— 逆向設計治好一個製程缺陷

> **這一課的形式**:整條流程已經寫成一支可直接執行的腳本
> (`scripts/09_toy_adjoint.py`,參考輸出見 [RESULTS.zh-TW.md](RESULTS.zh-TW.md))。
> 這份文件講**它在做什麼、為什麼可行**——想親手體驗,腳本每一段
> 都能單獨改參數重跑。

**先認幾個詞**(英文原詞就是 repo 程式碼和工具訊息裡會看到的那個字):

| 詞 | 這是什麼 |
|---|---|
| adjoint(伴隨法) | 一次正向加一次反向,就拿到全部參數的梯度。和線性代數的「伴隨矩陣」無關 |
| FOM(figure of merit) | 優化要最大化的目標值,這一課就是透射率;repo 程式碼一律寫 `FOM` |
| deterministic | 同一台機器、同一版本重跑,結果逐位元相同 |
| G2 閘門(G2 gate) | repo 制度化的檢查關卡;G2 專指「梯度先過有限差分比對才算數」那一關 |

## 這一課回答的問題

[第一課](../01-jax-port/README.zh-TW.md)結束時,E 場更新裡有一行 `/ eps`。
這一課問:**「每個格點的材料,對輸出各有多大影響?」**

暴力答案:一個格點挪一點、重跑一次模擬 → N 個參數要 N+1 次模擬。
設計區 20×20=400 個格點就是 401 次——不可行,這正是 2000 年代
光子設計只能「手調幾個參數」的原因(shift-or-shrink 的時代背景)。

adjoint 的答案:**一次正向 + 一次反向 = 全部 N 個參數的梯度**。
這是逆向設計成立的數學根基,也是 fdtdx/Meep 的 adjoint 在做的事。
JAX 把它自動化了:第一課的 `lax.scan` 是一個可微分程式,
`jax.grad` 對它做反向傳播,得到的就是 adjoint——不用手推 adjoint 場方程。

## 劇本(scripts/09_toy_adjoint.py)

1. **弄傷結構**:拿 phc_bend 的 90° 彎,模擬一個製程缺陷——打掉
   Layer-I 水平缺陷柱(點缺陷掃描裡傷害最大的那根;掃描本身見
   [docs/phc-bend-walkthrough.zh-TW.md](../../docs/phc-bend-walkthrough.zh-TW.md))。
   在這一課自己的設定下,少這一根就讓平均透射從 0.972 掉到 0.613。
2. **畫設計區**:缺陷周圍 2a×2a(20×20 格點),材料連續參數化:
   `eps = 1 + (eps_rod − 1) · sigmoid(θ)`,起點 = 受損結構本身。
3. **梯度驗證(G2 精神)**:隨機抽 3 個像素做中央有限差分,
   跟 `jax.grad` 逐一比對,相對誤差要 < 1e-5 才往下走。
   **不驗證的梯度不上桌**——這條紀律在本 repo 制度化成 G2 閘門
   ([src/invdx/gates/g2_gradcheck.py](../../src/invdx/gates/g2_gradcheck.py)),
   fdtdx 的梯度同樣要先過有限差分比對才算數。
4. **Adam 迭代**:讓梯度自己決定往設計區的哪裡放材料,
   看透射從受損值爬回去。

## 讀程式碼時的三個看點

- `simulate()` vs `run()`:可微分路徑必須全程留在 JAX 世界,
  一轉 numpy 梯度鏈就斷。這是「工程選擇服務物理需求」的實例。
- `objective = T 的平均`:你改成 `jnp.min`(最差頻率)就是 minimax
  ——本 repo 的優化器用的正是這個思想。它的光滑版是 `softmin`
  ([src/invdx/fab/filters_jax.py](../../src/invdx/fab/filters_jax.py)),
  [src/invdx/optimize.py](../../src/invdx/optimize.py) 用它把各波長的
  FOM 聚合成單一目標。
- 缺陷「治好」的方式不一定是把柱子原樣長回來——梯度只在乎透射,
  它找到的解可能長得完全不像原本的晶格。這就是逆向設計和
  「修回原狀」的本質差別。

## 親手玩的入口

```bash
PY=python   # 你的 invdx env 的 python
$PY scripts/09_toy_adjoint.py --gradcheck-only          # 只看梯度驗證
$PY scripts/09_toy_adjoint.py --tag mine                # 全流程
$PY scripts/09_toy_adjoint.py --iters 120 --lr 0.1      # 調優化器
```

改 `FSTARS`(目標頻率)、`DEFECT`(換一根柱打掉)、design_box 大小,
都會得到不同的「治療方案」——每次結果都存進 runs/,可以互相比。

## 連回大局

toy 引擎走到這裡,你擁有了一條**從零親手建立的完整逆向設計鏈**:
Yee 更新式 → 可微分模擬 → 驗證過的 adjoint 梯度 → 優化迴圈。
fdtdx 跑三維元件時,原理一模一樣,只是規模大一萬倍。

想再往下走,repo 裡有兩條現成的路:
[src/invdx/gates/g2_gradcheck.py](../../src/invdx/gates/g2_gradcheck.py)
是同一套梯度驗證用在 fdtdx 上的版本,可以拿 toy 的結果跟它對照;
[docs/phc-bend-walkthrough.zh-TW.md](../../docs/phc-bend-walkthrough.zh-TW.md)
則示範同一個結構在兩套引擎上的交叉驗證。
另外,toy 用的是一階 Mur 吸收邊界,它的殘餘反射是這條鏈上最粗的
近似——想更接近成熟求解器的數字,邊界條件是第一個該動的地方。
