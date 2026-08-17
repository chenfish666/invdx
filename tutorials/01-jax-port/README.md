# 第一課:把你的 FDTD 移植到 JAX(M-toy-2)

> **這一課的狀態**:物理主線的 JAX 引擎已完成並驗證
> (`src/invdx/toy/fdtd2d_jax.py`,與 numpy 版差 9e-16;實測產出見本資料夾
> [RESULTS.md](RESULTS.md))。這裡是**教學版**:挖空的骨架
> [fdtd2d_jax_skeleton.py](fdtd2d_jax_skeleton.py) 留給你親手填,
> 驗收時加 `--file` 指向它,不會動到主線。想自己練就**先別看** src 那份答案。

目標:親手把 `toy/fdtd2d.py`(numpy,~170 行,你已經用它重現過 PhC
90° 彎那個文獻經典基準)移植成 JAX 版,並證明**逐位元同物理**。這一課的產出不只是一個
更快的引擎——它是通往伴隨梯度的門票:JAX 版寫成後,`jax.grad` 就能
穿過整段時間演化,自動給你「透射率對任何設計參數」的梯度(第二課)。

寫程式的部分只有**三個空格**,全部在本資料夾的
[fdtd2d_jax_skeleton.py](fdtd2d_jax_skeleton.py),
每格對照 numpy 版的同名段落。鷹架(scan 迴圈、輸出打包)已就緒。

```bash
cd <invdx repo>
PY=python   # 你的 invdx env 的 python
```

---

## 第 0 步:概念(10 分鐘,先讀再動手)

### JAX 與 numpy 的唯一思想差異:陣列不可變

numpy 允許原地改:`Hx -= ...`、`Ez[1:-1] += ...`。
JAX 陣列**不可變**——每次「修改」其實是造一個新陣列:

| numpy(原地) | JAX(函數式) |
|---|---|
| `Hx -= a` | `Hx = Hx - a` |
| `Ez[1:-1,1:-1] += a` | `Ez = Ez.at[1:-1,1:-1].add(a)` |
| `Ez[0,:] = b` | `Ez = Ez.at[0,:].set(b)` |

為什麼要這樣?因為 JAX 的一切魔法(jit 編譯、自動微分、vmap)都
建立在「函數沒有副作用」上:輸入進、輸出出,中間不偷改任何東西。
編譯器因此能放心重排、融合、微分你的程式。
(別擔心效率:jit 編譯後 `.at[].add` 會被優化回原地操作。)

### lax.scan:帶狀態的迴圈

先跑這個三行範例,看懂再往下:

```bash
$PY scripts/08_toy_jax_lesson1.py --scan-demo
```

`scan(step, init, xs)`:`step(carry, x) -> (carry, y)` 被依序餵入
`xs` 的每個元素,狀態 `carry` 一路傳遞,每步的 `y` 自動疊成陣列。
對 FDTD 來說:

- `carry` = 場狀態 `(Ez, Hx, Hy)`
- `xs` = 每步的源振幅(整條波形先算好)
- `y` = 每步的探針讀值

scan 把整個時間迴圈編成**一個** XLA 程式——這是之後 `jax.grad`
能對整段演化求梯度的前提,也是它比 Python for 迴圈快的原因。

---

## 第 1 步:填空格 A —— H 場更新(法拉第定律)

打開 `fdtd2d_jax_skeleton.py` 找到空格 A,對照 numpy 版
[fdtd2d.py](../../src/invdx/toy/fdtd2d.py) 的「H from curl E」兩行,
改寫成不可變風格。把 `raise NotImplementedError` 那行刪掉。

自問:H 的更新需要 `.at[]` 嗎?為什麼 E 需要?
(提示:H 是整個陣列重算,E 只改內部切片。)

## 第 2 步:填空格 B —— E 場內部更新(安培定律)

對照「E interior from curl H」。兩件事別漏:
1. `Ez_old = Ez` 的留影已在鷹架裡、且在你的更新**之前**——想一想
   為什麼順序重要(Mur 要的是「上一步」的邊界值)。
2. `/ eps[1:-1, 1:-1]` ——材料唯一進場的位置。第二課我們就是對這個
   `eps` 求梯度,所以請對它保持敬意。

## 第 3 步:填空格 C —— Mur 吸收邊界(四條邊)

四行同一個模式,對照 numpy 版直接翻譯:

```
Ez = Ez.at[0, :].set( Ez_old[1, :] + mur * (Ez[1, :] - Ez_old[0, :]) )
```

(第一條邊直接送你,剩下三條自己來:`[-1,:]`、`[:,0]`、`[:,-1]`。)

## 第 4 步:驗收

```bash
$PY scripts/08_toy_jax_lesson1.py --file tutorials/01-jax-port/fdtd2d_jax_skeleton.py
```

通過長這樣(數字量級要對):

```
[diff] max|dE| = ~1e-15, max|dH| = ~1e-15 (場量級 ~0.4)
[PASS] 兩個引擎逐位元同物理
```

1e-15 = float64 機器精度:你的 JAX 引擎和 numpy 引擎是**同一個物理**,
不是「差不多」。這種等價證明就是 invdx 全專案的信任哲學,現在你
親手做了一次。

然後跑:

```bash
$PY scripts/08_toy_jax_lesson1.py --gpu
```

同一份程式碼、零修改,跑上 GPU——移植的第二個回報。

---

## 陷阱清單(卡住先看這裡)

- **float32 陷阱**:JAX 預設 float32,numpy 是 float64。驗收器已在
  import 最前面開 `jax_enable_x64`;如果你自己另寫測試腳本,這行
  必須在任何 jax 陣列誕生之前。差異卡在 1e-7 下不去?九成是這個。
- **在 step 裡放 Python 副作用**(print、append 到外面的 list):
  scan 只在「描圖」時執行你的 Python 一次,之後跑的是編譯產物——
  副作用不會每步發生。要記錄的東西一律走 `y` 輸出。
- **首跑很慢**:那是編譯(描圖 + XLA 最佳化),第二次才是真實速度。
  驗收器印了兩個時間,自己看差多少。
- **形狀錯**:curl 那行的切片形狀必須剛好 (nx-2, ny-2)。JAX 的報錯
  會告訴你形狀,對照 numpy 版切片一格一格對。

## 完成後

對答案:你的填法和主線 [src/invdx/toy/fdtd2d_jax.py](../../src/invdx/toy/fdtd2d_jax.py)
比一比——寫法可以不同,通過 1e-15 驗收就是同一個物理。
接著看第二課(tutorials/02):**第一個伴隨梯度**。
