[← back to docs index](README.md)

# 親手重現:光子晶體 90° 彎(文獻經典基準)

這份教學帶你**親手**重現光子晶體 90° 彎波導這個文獻裡的經典基準:
二維方晶格介電柱光子晶體、90° 彎波導、點缺陷對透射的影響。這類結構的
高透射彎角自 Mekis et al.(*"High Transmission through Sharp Bends in
Photonic Crystal Waveguides"*, Phys. Rev. Lett. 77, 3787, 1996)起就是
PhC 波導文獻中的標準教學題材。全程用本專案內建的兩個引擎——
`toy/`(你可以逐行讀懂的 ~120 行 FDTD)和 Meep(領域公認錨點)。

**基準參數**(全部寫在 `PhCBendConfig`,沒有任何數字藏在腳本裡):

| 參數 | 值 | 意義 |
|---|---|---|
| a | 1 µm | 晶格常數(程式內一切長度以 a 為單位) |
| R | 0.225a | 介電柱半徑 |
| ε | 10(n≈3.162) | 柱的介電常數 |
| 能隙 | f = 0.29–0.41 | 歸一化頻率 f = a/λ(文獻典型值,我們的錨點) |
| 偏振 | E 沿柱軸 | ⚠️ 見下方命名陷阱 |

> **命名陷阱(先讀這個)**:工程文獻常說打入「TE」光,但柱狀晶格的能隙
> 屬於 **E 平行柱軸** 的偏振——物理教科書(Joannopoulos, Johnson, Winn
> & Meade,*Photonic Crystals: Molding the Flow of Light*)稱之為
> **TM**,不少工程文獻反過來叫 TE。我們的 toy 引擎演化
> (Ez, Hx, Hy)正是 E 沿柱軸,所以偏振是對的;名字不要背,記「E
> 沿著柱子」就永遠不會錯。這是跨文獻比對時最常見的一種慣例地雷,
> 和 `engines/conventions.py` 記錄的 ½ 功率慣例是同一類故事。

每一步都是一條指令。預設參數 = 基準尺寸(21×21),單步約 30–60 秒;
想快速實驗就加 `--set n_side=11 --set toy_steps=3000`(數秒)。

```bash
cd <invdx repo>
PY=python   # 你的 invdx env 的 python
```

---

## 第 0 步:讀懂引擎本體(15 分鐘)

打開 [src/invdx/toy/fdtd2d.py](../src/invdx/toy/fdtd2d.py)。整個電磁學
就在三行更新式裡:

- `Hx -= (dt/dx)*(∂Ez/∂y)`、`Hy += (dt/dx)*(∂Ez/∂x)` —— 法拉第定律
- `Ez += (dt/dx)/eps * curl(H)` —— 安培定律;**材料唯一進場的位置**
  就是那個 `1/eps`

檢查點:為什麼 `eps` 只除在 E 更新、不動 H 更新?
(提示:非磁性材料 µ=1;介電質只改電位移 D=εE。)

## 第 1 步:看幾何

```bash
$PY scripts/06_phc_bend.py --stage eps --tag walkthrough
```

會印出三種佈局的 ASCII 圖(`o`=柱、`.`=空):`bulk`(能隙量測用的
8 週期板)、`straight`(直波導=歸一化參照)、`bend`(90° 彎)。

檢查點:彎的入口在左、出口在上;晶格比 n_side 多一圈(索引 −1 和
n_side)。那一圈是我們踩過的坑:沒有它,光會從晶體旁邊的真空走廊
繞過去,實測能隙從 −40 dB 被旁路洩漏淹沒成 −2 dB。**量測結構的
邊界設計和結構本身一樣重要。**

## 第 2 步:找能隙(對錨)

```bash
$PY scripts/06_phc_bend.py --stage gap --tag walkthrough
```

預期:阻帶(T < −20 dB,谷底約 −50 dB)落在 **f ≈ 0.27–0.41**。

對照文獻的 0.29–0.41 時有個物理細節值得搞懂:我們量的是**正入射
(Γ-X 方向)透射**,它的阻帶比「完整能隙」寬——完整能隙是所有
傳播方向阻帶的交集,下緣通常由 M 方向決定。所以正確的判準是
**我們的阻帶必須「包含」文獻的完整能隙**(✓ 0.27–0.41 ⊇
0.29–0.41),而不是邊緣逐點相等。兩個獨立引擎(toy 與 Meep)的
阻帶邊緣彼此吻合在 ~0.01 內,這比對上文獻的單一數字更有證據力
——conventions **教訓 6**(比形狀、不比單點)的又一次應用。

親手實驗(建議至少做一個):
- `--set eps_rod=8.9`(氧化鋁,Joannopoulos 教科書值):能隙應
  變窄並上移——介電對比越小,能隙越小。
- `--set r_rod=0.18`:柱變細,能隙位置與寬度怎麼動?
- `--set res_per_a=30`:離散化更細,下緣應更貼近文獻值。

## 第 3 步:90° 彎透射(基準主結果)

```bash
$PY scripts/06_phc_bend.py --stage bend --tag walkthrough
```

預期:**能隙內 T ≈ 0.85–1.1**——光被能隙禁止進入晶體,只能乖乖
沿著拔掉柱子的通道轉 90° 過彎,幾乎無損。這就是光子晶體波導的
魔法(Mekis et al. 1996 的高透射彎角),也是這個基準的核心結果。

檢查點:為什麼腳本只印能隙內的 T?能隙外那些 |T|>1 甚至負值是
什麼?(答:能隙外晶體不禁光,「波導」根本不導波,輸出線量到的
是四散雜訊的干涉,分子分母都沒有物理意義。知道**量測何時無效**
和知道怎麼量一樣重要。)

## 第 4 步:Meep 交叉驗證(信任的建立)

```bash
$PY scripts/06_phc_bend.py --stage meep --tag walkthrough     # ~數分鐘
$PY scripts/06_phc_bend.py --stage compare --tag walkthrough
```

同一份柱座標(`rod_centers_a`,兩引擎 import 同一個函式)進 Meep,
PML 吸收邊界、次像素平滑。預期:兩引擎能隙位置吻合、能隙內透射
曲線形狀一致;逐點數值有差(toy 是一階 Mur + 二值網格,Meep 是
PML + 平滑)——差異的**來源你都說得出名字**,這才叫交叉驗證。

## 第 5 步:點缺陷掃描(文獻的結論)

```bash
$PY scripts/06_phc_bend.py --stage defect --tag walkthrough   # ~4 分鐘
```

在彎角外側拔掉一根柱(Layer I = 最近層,II = 次層;水平/垂直/斜角
三種方位),看能隙內平均透射掉多少。文獻裡的常見結論:**垂直與水平
缺陷的影響遠大於斜角缺陷**。你的 delta 排序有沒有重現這個結論?

## 第 6 步(展望):從這裡到逆向設計

早期文獻的做法:人工挑幾個缺陷位置逐一模擬(或 shift-or-shrink
啟發式)。逆向設計的做法:把每根柱的位置/半徑當可微參數,伴隨法
一次算出全部梯度。同一個彎、同一個 FDTD,方法演進了二十年——
這就是從這個經典基準走向逆向設計的橋樑。要親手走完這一步,
接著讀 [`tutorials/01-jax-port`](../tutorials/01-jax-port/) (把 toy
引擎 JAX 化)與 [`tutorials/02-first-adjoint`](../tutorials/02-first-adjoint/)
(用有限差分驗證你的第一個伴隨梯度)。

---

## 附:每次跑完東西在哪

每條指令都經 `start_run` 建立 `runs/<時間戳>-phc-bend-walkthrough/`,
裡面有 `config.json`(含你的 --set 覆寫)、`cmdline.txt`、`env.txt`
與各階段的 JSON/npy 結果——半年後你仍能重建今天的每一張圖。
