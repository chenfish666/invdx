> [English](README.md) · **繁體中文**

# invdx

**給被動光子元件用的 GPU FDTD 工具箱,而且算出來的數字都核對得了。**

invdx 用 [fdtdx](https://github.com/ymahlau/fdtdx)(JAX、GPU、可微分)跑
FDTD(時域有限差分)模擬,再拿 [Meep](https://github.com/NanoComp/meep)
——一支獨立的 FDTD 程式,透過 subprocess bridge 呼叫——去核對這支引擎。從單元
測試到 fdtdx 對 Meep 的一致性,六道驗收關卡(gate)都過了,結果才算數;每個
run 目錄也都記下了設定、指令、環境和硬體。在這個基礎上,才是 adjoint(伴隨法)
梯度與逆向設計迴圈。**逆向設計是其中一項功能,不是整個工具箱。**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

## 它能做什麼

| | |
|---|---|
| **在 GPU 上模擬** | 在 fdtdx 上跑 quasi-2D(第三軸只有薄薄幾格、沒有設計自由度)與完整 3D 的模擬場域(scene);一次脈衝 run 就拿到整段頻譜;一條只做正向的快速路徑,比原版 fdtdx 快約 2 倍,場值位元完全相同 |
| **拿第二支引擎核對** | Meep 1.34(CPU、MPI),從原始碼建置,只當成一支獨立程式呼叫;兩支引擎之間的每一個慣例差異,都明文寫在同一個模組裡 |
| **驗證工具鏈本身** | G0–G5 六道關卡(見下表);缺了前置條件的關卡會直接失敗,不會被跳過 |
| **adjoint 梯度與逆向設計** | 穿過整段 FDTD 時間演化的梯度,並拿有限差分(FD)核對;一支 Adam 迴圈,任何可微分的參數化都能優化;密度 → conic filter → tanh projection,搭配 β 的 schedule;當掉可以續跑,Slurm requeue 也接得回去 |
| **重現一次 run** | 版本鎖死的環境(JAX/fdtdx 用 uv,Meep 用 spack);每個 run 目錄都記下設定、指令、環境和硬體探測結果 |

## 目前的成果

| 量了什麼 | 結果 |
|---|---|
| fdtdx 對 Meep 對解析的 Airy 公式(G5:介電平板、正入射) | T = 0.7441 / 0.7423 / 0.7398;fdtdx 與 Meep 差 0.24% |
| 穿過一整條設計路徑(filter → projection → 元件 → 模態重疊積分)的 adjoint 對有限差分(G2) | 三個 voxel 的相對誤差分別是 0.0008%、0.18%、0.07%(關卡門檻 5%) |
| 正向對反向激發(G4) | 兩者差 0.08 dB(關卡門檻 0.5 dB) |
| 快速正向路徑對原版 fdtdx | 8.0M cell 時 640 → 1245 Mcell-steps/s(1.94 倍);16.8M cell 時 653 → 1529(2.34 倍);場值位元完全相同 |
| 一張 24 GB 卡上跑得起來的最大正向 run | 原版 fdtdx 是 15.6M cell(到 27M 記憶體就爆掉,OOM);快速路徑加記憶體回收是 64M cell |
| adjoint 的記憶體 | 峰值 ≈ (372.18 + 268.0329·C) bytes/cell,C 是梯度重算用的 checkpoint 數(不是斷點續跑那種);斜率在 Turing 和 Ada 上一樣 |

表裡的 cell 即 voxel,也就是網格的一格。這裡全部都是模擬,沒有任何一項和實際
做出來的元件比對過。

### 這些成果是在什麼硬體上跑的

- GPU:NVIDIA Quadro RTX 6000、NVIDIA RTX 6000 Ada Generation
- CPU:AMD Ryzen 9 7950X、雙路 Intel Xeon Gold

## 裝得下什麼、裝不下什麼

三支引擎都是在均勻的直角網格上解 FDTD,而且每一支 adapter 給每種材料的,都是
一個實數、不隨頻率變的介電常數。題目的類別就由這件事決定。

**裝得下:** 被動、線性、無損的介電質結構,而且優化目標(FOM,figure of merit)
是兩次 run 的比值(有元件的那次 run 除以參考 run)。

**不動引擎就裝不下:** 有色散或有損耗的材料(所以金屬和電漿子都不行)、非線性
光學、增益與主動介質、非等向或磁光介質、隨時間變化的介電常數、熱或電的協同
模擬、非均勻網格、能帶結構(本徵模態)計算。

動手之前先知道這幾件事:

- **Meep 核對只到 2D 為止。** bridge 只有 2D 的量測工作,所以 3D 結構可以在
  fdtdx 上模擬、優化,但目前還不能拿 Meep 核對。缺的是一項 bridge 工作,不是
  物理上的限制。
- **端口需要一個寫得出來的模態。** 量測鏈裡沒有數值模態求解器,沒有閉式解的
  端口模態得自己提供(MPB 算得出來)。
- **一次 run 必須塞得進一張 GPU。** 用上面那個記憶體模型(GB = 10⁹ bytes),
  預算抓 16 GB(24 GB 卡上已知跑得起來的最大 adjoint run,峰值是 15.4 GB),
  C = 10 時大約裝得下 5.2M 個 cell:20 nm 格距下一個 3.5 µm 的立方體。fdtdx
  沒有真正的 2D,所以一個「2D」模擬場域仍帶著一條只有幾格厚的週期性第三軸,
  花費就是面內格數的那麼多倍。
- **逆向設計需要可微分的 FOM 和連續的設計變數。** 優化器接受任何能回傳 loss
  和它的梯度的參數化。像素網格上的密度直接就能用;幾何式的參數化要自己帶一支
  平滑的 rasterizer;離散的選擇則做不到。
- **toy 引擎是拿來學、拿來交叉核對的,不是拿來做設計的**:它用的是一階 Mur
  吸收邊界,不是 PML(perfectly matched layer)。

## 快速上手

```bash
bash scripts/bootstrap.sh   # 安裝並驗證 L1 層(uv、JAX、fdtdx)
make test                   # 單元測試,只要 CPU
make gates                  # 六道關卡全跑:G1-G4 要 GPU,G5 還要 Meep
```

Meep 是用 spack 從原始碼建的:[docs/env.zh-TW.md](docs/env.zh-TW.md)。

## 驗收關卡

關卡依序跑,每一道都假設前面的已經過了,每次都會寫出一份機器可讀的
`gates_report.json`。

| 關卡 | 檢查什麼 | 需要 |
|---|---|---|
| G0 unit | 整個 `tests/` 套件:重疊積分、filter、toy 物理、幾何、設計向量的來回轉換、斷點續跑用的 checkpoint | CPU |
| G1 api | 這個 repo 用到的 fdtdx API、JAX 看得到 GPU,以及裝了 Meep 時透過 bridge ping 得通 | GPU |
| G2 gradcheck | adjoint 梯度對有限差分:filter 的連鎖律、一個 toy 尺寸的模擬場域(scene),以及受測 problem 的設計路徑(5%) | GPU |
| G3 physics | 空的模擬場域裡的通量守恆(5%) | GPU |
| G4 reciprocity | 受測 problem 的正向對反向激發(0.5 dB);就是這道關卡在這裡抓到過一個差 2 倍的正規化 bug | GPU |
| G5 cross-engine | 在一塊介電平板上,fdtdx 對 Meep、也對解析的 Airy 公式(10%) | GPU、Meep |

關卡紅了就當成停線(整條流程停下,不往下走)。在 verification-and-validation
的意義上,這六道全部都是 *verification*:問的是程式有沒有解對它自己的方程式。
沒有一道是拿實際量測的元件去比的 *validation*。

關卡報告也會記下「這些數字是哪一個 problem 模組產出的」。這抓得到意外,例如一個
被複製過來卻沿用舊名字的模組,或一道報了數字卻說不出出處的關卡。它不是安全
邊界:problem 模組跑在同一個 process 裡,自己就能把報告寫出來。

## 整體怎麼組成

| 層 | 裝的是什麼 |
|---|---|
| A:引擎 | fdtdx(GPU,做設計)與 Meep(CPU,當參考),版本釘死、不改動。Meep 在 subprocess bridge 後面當成一支獨立程式執行,從來不被 import 進 invdx 的 process。釘住的版本擋路時,做法是 vendor 一個模組進來並標明出處,而不是去改套件本身 |
| B:方法層 | 設定與 run 目錄、關卡、設計用的 filter 與 projection、模態重疊積分、跨引擎慣例、problem 定義 |
| C:toy | 一支自己寫的 2D FDTD,和兩支引擎都不相干,用來學,也當第三方意見 |

## 踩出來的護欄(寫在 `engines/conventions.py` 裡)

1. Meep 的 DFT 場、|alpha|² 和通量都**沒有**含物理上那個 ½ 功率因子。自己寫
   的重疊積分必須明文把慣例接起來;不接的話,用它們算出來的任何功率比值都會差
   2 倍,而抓到它的是互易那道關卡。
2. `meep.adjoint` 的多頻梯度回來的形狀是 `(Nx, nf)`;要對 axis 1 加總。
3. 模擬解析度必須不低於設計網格密度,否則 adjoint 梯度會**系統性偏小**
   (實測:解析度 40 時偏低 5–8%,80 時準確)。
4. Meep 的 `decay_by` 預設值(1e-11)在同等精度下比 1e-6 慢約 3.4 倍,所以
   worker 一律明文把它傳進去。
5. **設定檔是唯一真相來源**;腳本永遠不硬寫數字。
6. 共振結構的跨引擎比對,比的是**頻譜峰形(ridge),絕不比單一波長的值**。
   兩支引擎會把同一個標稱幾何離散成稍微不同的等效線寬,而共振在波長上移動得比
   邊界誤差本身還快,所以兩支引擎可以在某個固定波長上差幾十 dB,峰位卻是
   吻合的。

## 文件

- [docs/README.zh-TW.md](docs/README.zh-TW.md):完整索引。
- [docs/new-problem.zh-TW.md](docs/new-problem.zh-TW.md):加入你自己的 problem。
- [docs/env.zh-TW.md](docs/env.zh-TW.md):uv 與 spack 兩層環境,以及怎麼從一份
  乾淨的 clone 重建。
- [tutorials/01-jax-port/](tutorials/01-jax-port/):把 toy 引擎移植到 JAX 的
  一堂課。

## 引擎授權

invdx 本身是 MIT。fdtdx 是 MIT;這裡 vendor 進來的那幾個 fdtdx 模組都標明了它的
出處。Meep 是 GPL-2.0-or-later,從來不被連結、不被 vendor、也不被 import 進
invdx 的 process:它在 subprocess bridge 後面當成一支獨立程式執行。Python
相依樹裡有三個 copyleft 套件,全都未經修改地使用:`tidy3d`
(LGPL-2.1-or-later,跟著 fdtdx 進來,跑在同一個 process 裡),以及 `certifi`
和 `tqdm`(MPL-2.0)。

## 引用

請透過 [`CITATION.cff`](CITATION.cff)(或 GitHub 的「Cite this repository」
按鈕)引用 invdx;物理引擎請直接引它們自己:FDTDX(Mahlau et al., Journal of
Open Source Software, 2026)與 Meep(Oskooi et al., Computer Physics
Communications, 2010)。
