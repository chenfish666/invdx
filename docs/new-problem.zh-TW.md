> [English](new-problem.md) · **繁體中文**

[← back to docs index](README.md)

# 怎麼加一個新的 problem

你想模擬一個這個 repo 從來沒看過的元件。這一頁是從「環境跑得起來」走到「我的
元件建好了、量到數字了、錨在一個已知答案上、有 gate 守著,而且——如果你要——
可以拿去優化」的那條路。

這是操作手冊,不是 API 參考。**每一步的結尾都有一條指令,告訴你這一步到底成了
沒有。** 全文示範用的那個 problem 是一個真的模組,短到可以整份印在下面,最後在
半分鐘的 CPU 時間裡和一個閉式解(closed-form,有公式可以直接算的解析答案)吻合
到 1%。

先決條件:`make check` 要過。過不了就先去 [`env.zh-TW.md`](env.zh-TW.md),不要
從這裡開始。

**先認幾個詞**(英文原詞就是你在程式碼、旗標和 log 裡會看到的那個字):

| 詞 | 這是什麼 |
|---|---|
| problem | invdx 裡一個完整的題目定義(幾何＋量測＋驗收),住在 `src/invdx/problems/`。全文的 `<your_problem>` 都是**你自己那一個**的代稱 |
| gate | 驗收關卡,代號 G0–G5,`make gates` 會依序跑完(`make check` 只跑 G0)。中文只用「關卡」補述,名字一律寫 gate |
| run | 一次執行,以及它落地的那個目錄(`runs/<timestamp>-…`) |
| driver | 這裡指「把設定、problem 模組和存檔串起來跑的那支腳本」,**不是**裝置驅動程式(device driver) |
| 模擬場域(scene) | FDTD 模擬的整個計算區域;`build_scene` 就是在組它 |
| voxel | 網格的一格。英文側寫 `cell` 的地方請一律讀成 voxel(**cell 即 voxel**),只有 `empty cell`、`outside the cell` 這種寫法指的是整個模擬場域 |
| rasterize | 把設計畫到設計像素網格上。**保留英文不譯**——這批文件通篇在講 grating(光柵),照字面譯出來的中文名一定和它撞在一起 |
| FOM(figure of merit) | 優化要最大化的目標量;本專案的慣例是 `loss = -FOM` |
| adjoint(伴隨法) | 一次正向加一次反向,就拿到全部設計參數的梯度。和線性代數的「伴隨矩陣」無關 |
| gradcheck | 拿有限差分(finite difference, FD)去核對 adjoint 梯度算得對不對的那道檢查 |

---

## 契約:一個 problem 模組到底是什麼

沒有抽象基底類別等你繼承,也沒有 entry-point 自動探索:**沒有任何東西會去掃描
problem**。一個模組要變成「叫得到」,只有兩條路——在
`src/invdx/problems/__init__.py` 的 registry dict 裡加一行,或是直接把一個帶點號
的模組路徑(dotted module path,例如 `yourpkg.problems.spiral`)餵給 `--problem`,
後者連這裡都不用改。除此之外,腳本和測試是**按名字 import 你的模組**;那仍然是
「為某一個 problem 而寫的程式碼」的接線方式,而 `scripts/` 底下大半都是這種。

你**必須**宣告的東西只有一件,而它之所以存在,是因為這一頁以前就停在那個洞上:
六道 gate 裡有兩道量的是一個具體的 problem,而新的 problem 以前預設**兩道都拿
不到**——而且是安靜地拿不到。所以一個 problem 模組的結尾長這樣:

```python
PROBLEM = ProblemSpec(
    config_cls=<YourName>Config,
    gradcheck_case=...,      # a factory, or Unsupported("why not")
    reciprocity_case=...,    # a factory, or Unsupported("why not")
)
```

兩個 gate 欄位都**沒有預設值**,所以「我忘了」會變成一個 import error,而不是
安靜地少掉兩道驗收。你把模組加進 `src/invdx/problems/__init__.py` 的 registry
dict 之後,`problems.load("<your_problem>")` 就會讀到這份宣告;`load` 一樣吃帶
點號的模組路徑,所以一個住在這個 repo 外面的 problem 不必先 vendor 進來也可以
被 gate 驗。確切的型別與細節在
[`src/invdx/problems/contract.py`](../src/invdx/problems/contract.py)。

**你的 problem 不替自己命名,而且你不必知道任何別人的名字才寫得出一個。**
`load` 被要求載入的那個字串**就是**名字——registry 的鍵,或是帶點號路徑的最後
一段(`yourpkg.problems.spiral` 的名字就是 `spiral`)——`load` 會把它蓋在回傳的
spec 上。這一頁的 `<your_problem>` 永遠是你自己取的那個名字的代稱,不是別人的
problem:底下沒有任何一步需要你去打聽一個不是你寫的 problem 叫什麼。把名字在
模組路徑和 registry 鍵之外**再寫第三次**,那是一份沒有推導關係、也沒有任何東西
在比對的手抄副本。

於是有兩條拒絕規則:

- 你硬要宣告 `name=`,而它和你載入時用的名字不一致 → `load` 直接丟例外,不會
  安靜地幫你改對。
- 帶點號路徑的最後一段撞到一個**已註冊**的 problem 名字(例如
  `yourpkg.problems.grating_coupler`)→ 在 import 之前就被擋掉,不管那個模組
  自己宣告了什麼、或什麼都沒宣告。

兩條擋的是同一件事:名字是 gate 報告歸檔用的鍵(`<your_problem>_f0`、
`<your_problem>_fd_checks`、`<your_problem>_sampling`、`details["problem"]`)。
名字錯了,你的數字就會頂著一個內建 problem 的標籤躺在 `gates_report.json` 裡。
兩條規則都碰不到「本來就有自己名字」的 problem——`spiral`、`mmi`、`tmm_stack`
從哪裡載入都載得起來。

`load` 同時會蓋上它解析出來的 import 路徑,gate 把它寫成
`details["problem_module"]`,所以報告說得出它的數字**打哪來**,而不只是說得出
它們叫什麼。名字是 loader 指派的;路徑則是唯一一個可以拿去對一棵真實檔案樹的
欄位,而且在外面那個 run 目錄(以及它的 `cmdline.txt`)消失之後,它仍然留在
`gates_report.json` 裡。

**`problem` 和 `problem_module` 這兩個鍵不是你寫的。** 兩個都由 gate 從 `load`
回傳的 spec 蓋上去,而 `ReciprocityCase.extra` 或 `GradcheckCase.info` 裡只要
出現這兩個名字之一,就會被拒絕,並且**把撞到的那個鍵印出來**。同一條拒絕也
涵蓋 gate 自己量的鍵和 runner 自己寫的鍵:

| 這一類鍵 | 例子 | 誰寫的 | 你放進 `extra` / `info` 會怎樣 |
|---|---|---|---|
| 身分 | `problem`、`problem_module` | gate 從 `load` 的結果蓋上 | 被拒絕,訊息點名撞到的那個鍵 |
| gate 自己量的數字 | `CE_fwd_dB`、`grad_max`、… | 該道 gate | 同上 |
| runner 寫的 | `seconds`、`reason`、`exception` | runner | 同上 |

**把你自己的數字放在你自己的名字底下。** 這條規則的理由是:安靜的合併會把你的
值歸檔到 gate 的名字底下——報告照樣 parse 得動,每一個該有的鍵都在,而**沒有
任何讀者分辨得出那個數字是誰的**。

spec 上那兩個欄位本身也有同一種要求:它們必須**恰好是 `str`**,不是「像 str
的東西」。因為 loader 對它們問的每一個問題(`str(...).strip()`、`==`)都是
**問那個值本身**,而一個 `str` 子類別會以自己的身分回答。這條規則一路延伸到
報告:`details` 裡**任何深度**的每一個鍵,以及 gate 蓋上去的那兩個身分值,都
必須恰好是 `str`——一個覆寫了 `__hash__` 的子類別,在任何一道護欄眼裡都不是它
拼出來的那個鍵,而 `json.dump` 還是會照那個名字把它寫出去。

身分鍵另外還有第二道檢查,而且**不依賴 gate 記得上面任何一件事**:寫報告之前,
runner 自己去看 `--problem` 到底要的是什麼,**只憑那個請求**推出身分(過程中
不載入任何東西),再拿去跟 gate 交出來的結果對。對不上(不管那道 gate 判成
什麼)、或是根本沒帶身分,這道 gate 就判 fail。

這道檢查**預設對每一道 gate 都開著**,包括明年某個從沒讀過這一頁的人寫的那一道。
免除只有兩種:

| 免除 | 條件 | 為什麼 |
|---|---|---|
| 結果已經是 `[FAIL]` | 那道 gate 本來就掛了 | 它自己的診斷不該被一句「出處不明」蓋掉 |
| 宣告 `MEASURES_PROBLEM = NoProblem(reason)` | 在 gate **自己的模組裡**,寫清楚它量的是什麼 | 有些 gate 真的不量任何 problem |

第二種長這樣:

```python
from invdx.gates import NoProblem      # 在套件內部要寫成:from .runner import NoProblem

MEASURES_PROBLEM = NoProblem(
    "G3 checks flux conservation in an EMPTY cell: ... there is no device "
    "in it to attribute `flux_in`, `flux_out` or their ratio to")
```

**你在 problem 模組裡寫什麼都關不掉它;你在 problem 模組裡漏寫什麼,一樣關不掉。**

確認這份宣告寫得成立的指令——它要建得起來,而空的理由建不起來:

```bash
uv run python -c "
from invdx.gates import NoProblem
print(NoProblem('G3 checks flux conservation in an EMPTY cell'))
try:
    NoProblem('')
except ValueError as e:
    print('empty reason refused:', str(e).split(':')[0])
"
```

理由是必填的,空的理由會像空的 `Unsupported(...)` 一樣丟例外。

這條規則的由來就是它的舊寫法:一個裸常數 `MEASURES_PROBLEM = False`。一次稽核
把 G3 的宣告**連註解區塊一起**抄進一道新的 gate,而那道新 gate 是真的在量一個
元件;於是它照樣報出數字、沒有蓋上任何身分、印了 `[ok]`。**三個字元在四個模組
裡都是對的,就不可能在第五個模組裡看起來是錯的**;但一段「講一個空 cell」的
句子可以,而且審查的人會看到它就躺在一堆耦合效率旁邊。這條規則買到的就只有
這個:**抄襲變得看得見,不是變得不可能**——和下一節〈這買到了什麼,又沒買到
什麼〉講的是同一條界線。它唯一真的強制到的事情是:`False` 現在 parse 不過去了,
所以沒有人會不小心把舊寫法帶著走。

`--problem` 是那個真相最常見的來源,但不是唯一的來源。`MEASURES_PROBLEM` 一共
三種形式:

| 形式 | 意思 | runner 怎麼做 |
|---|---|---|
| 不宣告(絕大多數 gate) | 我量的是 `--problem` 點名的那一個 | 從請求端推出身分,拿去和 gate 交出來的對 |
| `NoProblem("…")` | 我不量任何 problem,我量的是別的東西 | 跳過身分檢查,理由留在模組裡給人讀 |
| `'<name>'` | 不管誰點名,我永遠量這一個 | 用**同一套請求端的解法**去解那個名字,而不是去讀已經載入的 problem |

第三種的重點在最後一欄:所以**宣告裡寫的 problem,不可能和報告裡寫的是兩個不同
的東西**。它**不**驗證這道 gate 真的 import 了它點名的那個模組——一道嘴上說 A、
實際載入 B 去量的 gate,是在對自己的工作說謊,那是下一節那條界線的另一邊。

### 這買到了什麼,又沒買到什麼

這句話要講白,因為它會改變你讀「不是自己寫的那份報告」的方式:**上面這些規則
是記帳,不是安全邊界。** 載入你的 problem 就是 import 它,而一個被 import 的
模組和 gate 跑在同一個 process 裡(這裡的 process 指作業系統裡一個獨立執行的
程式實例)。它可以直接伸手進 `invdx.gates`、把 runner 的函式換掉,或是根本不
模擬就自己寫一份 `gates_report.json`。一個**存心說謊**的模組做得出和誠實報告
一個位元組都不差(byte-identical)的報告——這個 repo 的一次稽核就做了一個:純 CPU 的替身,在本來該是
91 秒 GPU run 的地方放一個 `time.sleep`——而任何跑在同一個 process 裡的檢查都
抓不到它。problem 模組如果是故意說謊,報告就不是證據,這裡沒有任何東西改變得了
這件事。

這些規則抓的是**沒有人存心要它發生**的那一整片東西,而你在寫新 problem 的時候,
待的正是那一片:忘了宣告一道 gate、抄了一個模組又留著它原本的名字、把檔案取成
和內建 problem 一樣的名字、讓你的 `extra` dict 蓋到 gate 的鍵上、寫一道報得出
數字卻沒有出處的 gate。這幾件事以前每一件都會產出一份綠色的報告,現在每一件都
會產出一個大聲的失敗,並且點名是哪個鍵、該怎麼修。

界線停在哪裡,也一起講清楚,你才知道一份報告的哪一半有機器在背書:

- runner 只重推得出**那兩個身分鍵**,所以它自己主動檢查的也只有那兩個。
- 你的 `extra` 和 `info` 不會蓋掉 gate 量出來的數字,靠的是撞鍵拒絕,而那道拒絕
  住在 `runner.merge_problem_dict` 裡。**一道沒有把你的 dict 走過那個函式的
  gate**(這個 repo 內建的都有走)會讓你的值蓋在它自己的值上面,而下游沒有任何
  東西分辨得出來。那是寫下一道 gate 的人要守的規矩,不是這一頁給得起你的保證。
- 出處欄位是讓一份報告**變得可查**的東西:它給讀者一條模組路徑,叫他自己去讀。
  **它是通往證據的路標,不是證據本身。**

其他的都是慣例,不是契約,而且誠實的理由值得寫出來:兩個內建 problem 的模組層級
函式名字,交集是**空的**。`grating_coupler` 和 `phc_bend` 沒有共用任何一個可呼叫
的東西。所以下面這張表描述的是你最後會寫出來的**形狀**,不是任何東西會去 import
的名字。

| 你提供什麼 | 誰會用它 | 一定要嗎? |
|---|---|---|
| `PROBLEM = ProblemSpec(...)` —— 一個 config 類別,加上每一道「量 problem」的 gate 各一個答案;不含名字,`load` 自己推 | `problems.load`、gate G2 Part C 與 G4 | 要 |
| 一個 `@dataclass` config,繼承 `config.BaseConfig` | `cli.apply_overrides`(`--set`)、`cli.start_run`(寫 `config.json`) | 要 |
| 幾何,寫成單純的 numpy / 單純的資料 | 你自己的 scene builder、你自己的測試、`invdx.viz` | 實務上要 |
| 每個你用到的引擎各一個 scene builder | 你自己的量測函式 | 一個引擎一個 |
| 量測函式,回傳單純可 JSON 化的 dict | driver 腳本、gate、`runio.save_json` | 要 |
| `vg_fn(p, beta) -> (loss, grad)`,其中 `loss = -FOM` | `optimize.run_loop`、`ProblemSpec.gradcheck_case` | 只有做逆向設計才要 |

**一個能動的最小 problem:一個 config 子類別、一個幾何函式、一個「兩次 run 相除」
的量測,以及一份把兩道 gate 都回答掉的 `PROBLEM` 宣告——就算兩個答案都是
`Unsupported` 也算數。** 第二個引擎、可微分的 FOM、優化用的 driver 全部是選配,
而且每一項都是另外一天的工作量。不要從它們開始。

最小的完整範例是
[`tests/fixture_problems/tmm_stack.py`](../tests/fixture_problems/tmm_stack.py):
不用引擎、不用 GPU,而且兩道 gate 都拿得到。它住在 `tests/` 而不是 `problems/`
底下,因為它是契約的測試夾具,不是誰真的要設計的元件。

---

## 第 0 步:決定要抄哪一份

| 從哪抄 | 什麼時候 | 為什麼 |
|---|---|---|
| [`src/invdx/problems/phc_bend.py`](../src/invdx/problems/phc_bend.py) | 你的 problem 跑在 toy 二維引擎和/或 Meep 上,用 CPU | 純 numpy——它不 import jax 也不 import fdtdx,正是這一點讓 `engines/meep_worker.py` 可以在 Meep 環境**裡面**把它 import 進去,於是兩個引擎吃的是同一份幾何定義。config ＋ 幾何 ＋ 幾個相除的量測,沒有別的東西。 |
| [`src/invdx/problems/grating_coupler.py`](../src/invdx/problems/grating_coupler.py) | 你的 problem 需要 fdtdx 的 GPU 引擎和/或 adjoint 梯度 | 整個 repo 裡最長的模組,而那個長度大半是**一個**元件的量測鏈:不要抄整份檔案,抄段落。 |

要學一個 problem 模組的**形狀**,`phc_bend` 是比較好的範本;`grating_coupler`
則是「一條 fdtdx 量測鏈長什麼樣」的參考。動手寫之前值得先讀的是這幾段:

| `grating_coupler.py` 的段落 | 它示範什麼 |
|---|---|
| `build_scene` | 怎麼把一串 fdtdx 物件加上擺放約束組起來 |
| `_run`、`_phasor` | 怎麼跑一個模擬場域,並把 `PhasorDetector` 的一條線讀回來 |
| `characterize`、`beam_power_and_tilt` | 一個量測,以及它配套的歸一化 run |
| `_box_bounds`、`check_energy_closure` | 怎麼寫一道**寧可拒絕、也不回傳一個看起來合理的數字**的護欄 |
| `design_device`、`build_scene_design` | 可微分的 `fdtdx.Device` 那條路 |
| `te0_target_on_monitor`、`ce_from_arrays`、`make_ce_value_and_grad` | 一個被 trace 的 FOM,以及它的 value-and-grad factory |

這一頁接下來會照 `phc_bend` 的風格從零建一個新的 problem,並且在每一步指出
`grating_coupler` 對應的地方在哪。

---

## 第 1 步:config 子類別

建 `src/invdx/problems/<your_problem>.py`,從 config 開始寫。**任何可調的東西
都不准住在別的地方**:腳本永遠不硬寫數字,因為 `cli.start_run` 會把
`config.json` 快照下來,而幾個月後讓一次 run 重現得了的正是那份快照。

```python
from dataclasses import dataclass

import numpy as np

from ..config import BaseConfig


@dataclass
class SlabConfig(BaseConfig):
    # ---- 幾何(um) ----
    n_slab: float = 2.0
    t_slab: float = 0.5

    # ---- 頻帶 ----
    lam_min: float = 1.0
    lam_max: float = 2.5
    n_freq: int = 31

    # ---- 數值(toy 引擎) ----
    cells_per_um: int = 40
    pad_um: float = 3.0         # 平板前後的真空
    height_um: float = 12.0     # 模擬場域的橫向範圍
    toy_steps: int = 6000
    toy_courant: float = 0.5

    @property
    def freqs(self):
        return np.linspace(1.0 / self.lam_max, 1.0 / self.lam_min, self.n_freq)
```

四條會咬人的規則:

1. **每個欄位都要有預設值。** `BaseConfig` 的欄位全都有預設值,所以 dataclass
   的繼承會直接拒絕任何沒有預設值的欄位。
2. **浮點數的預設值要寫成浮點字面值。** `cli._cast_like` 會把 `--set` 傳進來的
   值轉成**當下那個預設值**的型別,所以一個宣告成 `0` 的欄位永遠是 int,而
   `--set w=0.3` 會死在 `int("0.3")` 裡。`GratingCouplerConfig.w_s11` 就是為了
   這個理由把警告寫在欄位旁邊。
3. **推導出來的值一律用 `@property`,絕不寫成欄位。** property 不會出現在
   `dataclasses.asdict` 裡,所以 `config.json` 保持是「重建一次 run 所需的最小
   輸入集合」。`PhCBendConfig.freqs` 和 `GratingCouplerConfig.X0` / `cell_x` /
   `cell_z` 就是這個寫法。
4. **你根本不用的 `BaseConfig` 欄位還是會出現**在 `config.json` 和「合法鍵」的
   錯誤訊息裡。那是可以接受的雜訊,不是要修的 bug:`PhCBendConfig` 繼承了
   `min_feature`、`eta_e` 和 `beta_schedule`,一個都沒讀。

**確認它成了。** 一個能來回轉換、而且會拒絕胡說八道的 config:

```bash
uv run python -c "
from dataclasses import asdict
from invdx.problems.slab import SlabConfig
cfg = SlabConfig(n_slab=3.0)
print(len(asdict(cfg)), 'fields'); print(cfg.freqs[:3])
"
```

等你有了 driver 腳本(第 8 步)之後,再確認 `--set` 這條路:

```
$ uv run python scripts/22_slab.py --set nonsense=1
unknown config key: nonsense
  valid keys: beta_schedule, cells_per_um, design_grid_per_um, ...
```

如果打錯的鍵是被安靜忽略掉的,那就表示你沒有走 `cli.apply_overrides`。

> **這是給呼叫端傳的值,不是給使用者調的。** `grating_coupler` 會在組模擬場域
> 之前設 `cfg._lams_um`,再用 `finally` 還原。不是 dataclass 欄位的屬性,對
> `config.json` 和 `--set` 都是隱形的——給呼叫端在程式裡設是對的,拿來放任何
> 使用者應該調得到的東西就是錯的。

---

## 第 2 步:先做幾何,而且在模擬之前先看一眼

**在你親眼看過程式實際建出來的介電常數分布之前,不要開始模擬。** 兩個內建的
problem 都把這件事做成正式的一個 stage:`python scripts/06_phc_bend.py --stage eps`
會把三種佈局都印成 ASCII 圖,並把陣列存起來。

對一個 numpy/toy 的 problem 來說,幾何就只是一個陣列:

```python
def epsilon_grid(cfg, layout):
    """Rasterized permittivity. layout "empty" is the normalization run.

    The outermost cells stay vacuum on purpose: the toy engine's first-order
    Mur boundary assumes the vacuum wave speed there (toy/fdtd2d.py).
    """
    nx = int(round((2 * cfg.pad_um + cfg.t_slab) * cfg.cells_per_um))
    ny = int(round(cfg.height_um * cfg.cells_per_um))
    eps = np.ones((nx, ny))
    if layout == "empty":
        return eps
    i0 = int(round(cfg.pad_um * cfg.cells_per_um))
    i1 = i0 + int(round(cfg.t_slab * cfg.cells_per_um))
    edge = max(2, cfg.cells_per_um // 4)
    eps[i0:i1, edge:ny - edge] = cfg.n_slab ** 2
    return eps
```

```bash
uv run python -c "
import numpy as np
from invdx.problems import slab
cfg = slab.SlabConfig()
eps = slab.epsilon_grid(cfg, 'slab')
print(eps.shape, np.unique(eps))
print('slab thickness in cells:', int((eps.max(axis=1) > 1).sum()))
"
```

**fdtdx 的 problem 也有同一種檢查**,而且值得花那兩分鐘。`build_scene` 回傳
`(sim_config, object_list, constraints)`,這時候還沒有任何東西被擺定位;
`fdtdx.place_objects` 把約束解掉之後,網格就讀得回來了:

```bash
uv run python -c "
import jax, numpy as np, fdtdx
from invdx.problems import grating_coupler
cfg = grating_coupler.GratingCouplerConfig(spacing_um=0.040)
sim_config, objs, cons = grating_coupler.build_scene(
    cfg, teeth=grating_coupler.uniform_grating_teeth(cfg, 0.6, 0.5))
objects, arrays, params, sim_config, _ = fdtdx.place_objects(
    object_list=objs, config=sim_config, constraints=cons,
    key=jax.random.PRNGKey(0))
eps = 1.0 / np.asarray(arrays.inv_permittivities)[0]
print('eps grid:', eps.shape, 'unique:', np.unique(np.round(eps, 3)))
print('time steps:', sim_config.time_steps_total)
print('wg_mon phasor shape:', arrays.detector_states['wg_mon']['phasor'].shape)
"
```

```
eps grid: (500, 4, 243) unique: [ 1.     2.094 12.271]
time steps: 19669
wg_mon phasor shape: (1, 1, 2, 1, 4, 62)
```

這只花幾秒鐘,而且是在你付出一次 run 的代價**之前**就抓到:一個被擺到模擬場域
外面的方塊、一個吸附之後變成零個 voxel 的特徵、一個朝向錯掉的偵測面、一個
默默橫跨了你本來想留空的那個軸的物件、一個大小是你預期十倍的網格。

有兩個約束要在設計時就繞開:

- **toy 引擎:** 最外圈的 `eps` 必須是 `1.0`。一階 Mur 邊界假設那裡是真空波速。
  材料碰到邊界**不會**丟例外——它會反射,而那個反射就決定了你的 noise floor
  (雜訊底線)。
- **邊界那一圈是量測的一部分。** `phc_bend` 特地把多一圈晶格位置帶**進**邊框裡;
  少了它,光會從晶體周圍的真空繞過去,量到的能隙內壓抑就變成「被繞道限制」而不是
  「被晶體限制」(這個效應值好幾十 dB——見 `rod_sites` 的 docstring,以及
  [`phc-bend-walkthrough.zh-TW.md`](phc-bend-walkthrough.zh-TW.md) 的第 1 步)。

---

## 第 3 步:量比值,而且要有一次一模一樣的歸一化 run

這個 repo 裡每一個物理數字都是**兩次 run 相除**,而那兩次 run 只差在被測的那
一件事上:

- `phc_bend`:彎的輸出 / 直波導的輸出;晶體板 / 空的模擬場域(scene;英文原文寫 `empty cell`,
  依裁定這裡的 cell 指整個 scene,不是網格的一格)
- `grating_coupler`:波導監測面上的模態功率 / 空模擬場域 run 量到的入射光束功率

**「只差在那一件事」是照字面的意思。** `grating_coupler` 的模組 docstring 寫明
了讓它的比值成立的條件:所有的組成塊都是從**同一個波長、同一個 run 時長**的
偵測器場算出來的,所以 phasor 的縮放因子會**恰好**抵消。
`GratingCouplerConfig.sim_time_s` 在欄位旁邊又講了一次:量測 run 和它的歸一化
run 之間,這個值必須**一模一樣**。只改其中一邊,你會得到一個錯的答案,而且沒有
任何錯誤訊息。

歸一化 run 的檢查清單——同樣的網格間距與形狀、同樣的光源、同樣的步數 /
`sim_time_s`、同樣的偵測面、同樣的波長清單。**結構是唯一的差別。**

量測函式本身應該回傳**單純可 JSON 化的 dict**(list、float、字串——不要 numpy
純量,不要陣列)。這正是 `runio.save_json` 能把它們直接寫進 run 目錄的原因,
也是 `toy_bend_transmission` 結尾都是 `.tolist()` 呼叫的原因。

---

## 第 4 步:錨到一個你沒有去湊的答案上

一個只跟自己吻合的量測什麼都證明不了。這個 problem 值得拿去優化之前,它需要
一個**錨**——一個不是你挑的數字。

| 錨 | 成本 | 這個 repo 裡的例子 |
|---|---|---|
| 閉式解析解 | 幾秒 | `gates/g5_crossengine.analytic_transmission`(Airy 平板);`grating_coupler.slab_te0_neff`(非對稱平板的色散關係) |
| 一個文獻值 | 幾分鐘 | `phc_bend` 的參考能隙 `f = 0.29..0.41`,`scripts/06_phc_bend.py` 裡的 `GAP_REF` |
| 第二個引擎 | 幾分鐘到幾小時 | `phc_bend.meep_bend_transmission`,經由 `engines/meep_bridge.py` |

你的元件有哪個錨就挑最便宜的那個,而且**把它以函式的形式留在 repo 裡,不要
留成 commit message 裡的一個數字**。

### 完整範例,從頭到尾

下面是一個完整的新 problem——`src/invdx/problems/slab.py`,一個由你建立的
檔案——它把第 1 到第 4 步全部做完。內容是正入射通過一片無損耗介電平板的透射,
跑在 toy 引擎上,拿 Airy 公式對答案。

```python
"""Normal-incidence transmission of a lossless dielectric slab in air.

Units: lengths in um, frequencies in 1/um (f = 1/lambda).
Engine: the self-written 2D toy FDTD (CPU only). Every reported number is a
ratio of two runs that differ only by the slab, so the absolute source
amplitude cancels. Anchor: the Airy formula for a lossless slab, which
contains no fitted parameter.
"""

from dataclasses import dataclass

import numpy as np

from ..config import BaseConfig


@dataclass
class SlabConfig(BaseConfig):
    # ---- 幾何(um) ----
    n_slab: float = 2.0
    t_slab: float = 0.5

    # ---- 頻帶 ----
    lam_min: float = 1.0
    lam_max: float = 2.5
    n_freq: int = 31

    # ---- 數值(toy 引擎) ----
    cells_per_um: int = 40
    pad_um: float = 3.0         # 平板前後的真空
    height_um: float = 12.0     # 模擬場域的橫向範圍
    toy_steps: int = 6000
    toy_courant: float = 0.5

    @property
    def freqs(self):
        return np.linspace(1.0 / self.lam_max, 1.0 / self.lam_min, self.n_freq)


def epsilon_grid(cfg, layout):
    """Rasterized permittivity. layout "empty" is the normalization run.

    The outermost cells stay vacuum on purpose: the toy engine's first-order
    Mur boundary assumes the vacuum wave speed there (toy/fdtd2d.py).
    """
    nx = int(round((2 * cfg.pad_um + cfg.t_slab) * cfg.cells_per_um))
    ny = int(round(cfg.height_um * cfg.cells_per_um))
    eps = np.ones((nx, ny))
    if layout == "empty":
        return eps
    i0 = int(round(cfg.pad_um * cfg.cells_per_um))
    i1 = i0 + int(round(cfg.t_slab * cfg.cells_per_um))
    edge = max(2, cfg.cells_per_um // 4)
    eps[i0:i1, edge:ny - edge] = cfg.n_slab ** 2
    return eps


def _ports(cfg):
    """Source line and flux line. Both sit in vacuum and are IDENTICAL in the
    measurement and the normalization run — that identity is what makes the
    ratio mean anything."""
    nx = int(round((2 * cfg.pad_um + cfg.t_slab) * cfg.cells_per_um))
    ny = int(round(cfg.height_um * cfg.cells_per_um))
    edge = max(2, cfg.cells_per_um // 4)
    return {"src": {"i": int(round(1.0 * cfg.cells_per_um)),
                    "j0": edge, "j1": ny - edge},
            "out": ("x", nx - int(round(1.0 * cfg.cells_per_um)),
                    edge, ny - edge)}


def _run(cfg, layout):
    from ..toy import fdtd2d

    eps = epsilon_grid(cfg, layout)
    ports = _ports(cfg)
    dx = 1.0 / cfg.cells_per_um
    fcen = 0.5 * (cfg.freqs[0] + cfg.freqs[-1])
    spread = 1.0 / (np.pi * (cfg.freqs[-1] - cfg.freqs[0]) / 2)
    out = fdtd2d.run(
        nx=eps.shape[0], ny=eps.shape[1], dx=dx, steps=cfg.toy_steps,
        source={**ports["src"], "t0": 4 * spread, "spread": spread,
                "fcen": fcen},
        eps=eps, courant=cfg.toy_courant,
        line_probes={"out": ports["out"]})
    dt = cfg.toy_courant * dx
    return fdtd2d.line_flux_spectrum(out["lines"]["out"], cfg.freqs, dt, dx,
                                     sign=-1.0)


def toy_transmission(cfg):
    """T(f) = P(cell with slab) / P(empty cell). Plain JSON-able dict."""
    p_empty = _run(cfg, "empty")
    p_slab = _run(cfg, "slab")
    return {"freqs": cfg.freqs.tolist(), "T": (p_slab / p_empty).tolist()}


def analytic_transmission(cfg):
    """Airy transmission of a lossless slab in air — the parameter-free
    anchor. Same formula as gates/g5_crossengine.analytic_transmission."""
    n, t = cfg.n_slab, cfg.t_slab
    s = np.sin(2 * np.pi * n * t * cfg.freqs) ** 2
    return 1.0 / (1.0 + ((n ** 2 - 1) ** 2 / (4 * n ** 2)) * s)
```

`fdtd2d.run` 有兩件事要注意,而且兩件各自都是會安靜給錯答案的陷阱:光源那個
dict 帶了 `fcen`,少了它脈衝就變成基頻的(baseband),頻譜一離開 DC 就死掉;
以及 `line_flux_spectrum` 需要一個 `sign`,因為沿 `+x` 的功率流是 `Sx = -Ez*Hy`,
沿 `+y` 的卻是 `Sy = +Ez*Hx`。兩件都寫在
[`toy/fdtd2d.py`](../src/invdx/toy/fdtd2d.py) 各自的定義旁邊。

**確認它成了**——這正是那個錨存在的意義:

```bash
uv run python -c "
import numpy as np
from invdx.problems import slab
cfg = slab.SlabConfig()
T = np.array(slab.toy_transmission(cfg)['T'])
Ta = slab.analytic_transmission(cfg)
print('max rel err vs Airy:', float(np.max(np.abs(T - Ta) / Ta)))
"
```

```
max rel err vs Airy: 0.010295862885515091
```

CPU 上大約 30 秒。如果你的數字是 0.5 而不是 0.01,常見成因依序是:歸一化 run
和量測 run 不一模一樣、flux 的正負號、光源沒有載波頻率、材料碰到了 Mur 邊界。

---

## 第 5 步:會安靜出錯的那些慣例

**這一節要讀兩遍。** 下面每一條都會給你一個看起來很合理的數字,而且不報錯。
它們被收成一條條可執行的規則,放在
[`engines/conventions.py`](../src/invdx/engines/conventions.py) 裡,就是為了讓
一個新的 problem 不必再重新發現一遍。

| 陷阱 | 踩到的時候看起來像什麼 | 該用什麼 |
|---|---|---|
| Meep 在 DFT 場、`\|alpha\|²` 和 flux 上都略掉了物理上的那個 ½ | 一個乾淨、一致的 2 倍(3 dB)偏差,任何自洽性檢查都看不見 | `conventions.MEEP_POWER_OMITS_HALF`、`conventions.meep_to_physical_power` |
| 模擬解析度低於設計網格的密度 | 正向場看起來是對的;adjoint 梯度卻系統性偏小(res 40 實測低 5–8%) | `conventions.assert_resolution_covers_design_grid(cfg)`;走 fdtdx Device 那條路的話再加 `grating_coupler.assert_design_grid_snaps(cfg)` |
| minimax FOM 的波長取樣太稀 | 被抽到的那幾個點每次迭代都在進步,而它們之間的頻譜整個塌下去;優化器自己的讀數變得沒有意義 | `conventions.assert_fom_sampling_covers_band(spacing_nm, feature_nm)` |
| 拿單一固定波長去比兩個引擎 | 兩個引擎的頻譜峰位其實吻合,單點卻差幾十 dB——不同的離散化給出不同的**有效**幾何,每一條邊最多差半個 voxel | `conventions.CROSS_ENGINE_COMPARE_SPECTRA`:比曲線、比峰位、比峰值 |
| `meep.adjoint` 的多頻梯度回來是 `(Nx, nf)` | 直接 ravel 會得到一個**長度**就錯的梯度,而它照樣跑得動 | `conventions.collapse_multifreq_gradient(dJ)`(對頻率求和) |
| Meep 的 `decay_by` 預設值(1e-11) | 正確,而且在同等準確度下比 1e-6 慢約 3.4 倍 | 一律像 `phc_bend.meep_payload` 那樣明確傳 `cfg.dft_decay_tol` |
| 對一個之後要進到加總裡的 flux 取 `abs()` | 取絕對值對**比值**是對的,對**守恆檢查**是錯的:各個面的 flux 只有在流入與流出保持相反符號時才可能互相抵消 | `grating_coupler.phasor_line_power`(取絕對值,給比值用)對上 `grating_coupler.signed_poynting_flux_x` / `signed_poynting_flux_z`(帶符號,給加總用) |
| 把瞬時的時域偵測器和 phasor 量混在一起 | 一個量級看起來合理、意思卻不存在的數字——早期那份能量帳加起來變成 144–151% 就是這樣來的 | 一種量只用一個偵測器家族;`grating_coupler.energy_budget` 的 judgment #1 把這件事寫清楚了 |
| 把一個承載設計自由度的軸平均掉 | run 正常結束、數字看起來很普通,而優化器拿到的是一個被抹平的目標函數 | `grating_coupler.ce_from_arrays` 會在監測面的 `ny` 和 `cfg.n_y_cells` 對不上時丟例外;照著那個形狀寫一道檢查 |

**怎麼讓你自己的陷阱大聲起來。** 這個 repo 用的寫法是:一個**在昂貴的工作開始
之前就丟例外**的函式,放在 problem 模組裡,而且把**理由**寫進例外訊息裡:

- `grating_coupler._box_bounds` 會在幾何違反了「盒子裡沒有損耗、也沒有光源」這個
  前提時,拒絕報出能量閉合檢查的結果,並且說出該改哪一個參數。
- `grating_coupler.assert_design_grid_snaps` 會拒絕一個沒辦法同時整除層厚與設計
  像素的網格間距,並且把可以的那幾個間距列出來。
- `grating_coupler.energy_budget` 第一行就呼叫 `_box_bounds`,刻意讓它在**付出
  一次模擬的代價之前**就丟例外。

如果你的 problem 在設定錯誤時會產生一個數字而不是一個錯誤,現在就把那道護欄
寫下來。這比之後在結果裡發現它便宜太多了。

---

## 第 6 步:測試,G0 免費幫你收走

測試放在 `tests/test_<your_problem>.py`。**不需要註冊任何東西**:
[`gates/g0_unit.py`](../src/invdx/gates/g0_unit.py) 會對整個 `tests/` 目錄跑
pytest,所以一個丟進去的檔案從下一次 run 起就在那道 gate 裡了。

先測那些**完全不需要模擬**的事情——它們才是能在幾毫秒內抓到幾何錯誤的那些。
[`tests/test_phc_bend.py`](../tests/test_phc_bend.py) 是範本:每種佈局的柱子
數量、介電常數值與面積佔比、幾何必須滿足的一個對稱性,以及「相除得到量測」的
那兩次 run 的光源到監測面路徑長度相等。然後才是一個用粗設定跑的快速物理回歸。

```python
"""Pure-math tests for the slab problem, plus one fast physics regression."""

import numpy as np

from invdx.problems import slab

CFG = slab.SlabConfig(cells_per_um=20, height_um=6.0)


def test_epsilon_grid_binary_and_placed():
    eps = slab.epsilon_grid(CFG, "slab")
    assert set(np.unique(eps)) == {1.0, CFG.n_slab ** 2}
    assert np.all(slab.epsilon_grid(CFG, "empty") == 1.0)
    # Mur 邊界假設最外圈是真空
    assert eps[0].max() == 1.0 and eps[-1].max() == 1.0
    assert eps[:, 0].max() == 1.0 and eps[:, -1].max() == 1.0


def test_slab_thickness_in_cells():
    eps = slab.epsilon_grid(CFG, "slab")
    assert (eps.max(axis=1) > 1.0).sum() == round(CFG.t_slab * CFG.cells_per_um)


def test_analytic_peaks_at_half_wave():
    # 2*n*t*f 剛好是整數時 T 恰為 1(半波平板)
    cfg = slab.SlabConfig(lam_min=1.0, lam_max=1.0, n_freq=1)
    cfg.n_slab, cfg.t_slab = 2.0, 0.25          # f = 1 時 2*n*t*f = 1
    assert abs(slab.analytic_transmission(cfg)[0] - 1.0) < 1e-12


def test_toy_matches_airy():
    # 快速物理回歸(約 20 秒):量到的曲線必須在整個頻帶上都貼著
    # 那個沒有任何擬合參數的解析錨
    cfg = slab.SlabConfig(n_freq=11, toy_steps=4000)
    Ta = slab.analytic_transmission(cfg)
    T = np.array(slab.toy_transmission(cfg)["T"])
    assert np.max(np.abs(T - Ta) / Ta) < 0.05
```

**確認它成了:**

```bash
uv run python -m pytest tests/test_slab.py -q      # 只跑你這一個檔案
make check                                          # G0:整套測試
```

把物理回歸的時間壓在**秒**的量級,不要到分鐘。`make check` 是會被一直跑的東西,
而一道很慢的 gate 就是一道大家會開始跳過的 gate。

---

## 第 7 步:繼承現成的 gate,以及自己加一道

先弄清楚一個新的 problem 繼承得到什麼。六道內建 gate 裡有四道與 problem 無關;
另外兩道量的是 `--problem` 點名的那個 problem,而你要拿到它們就得宣告一個 case:

| gate | 對一個新的 problem 而言 |
|---|---|
| G0 `unit` | 免費,而且你一加測試,它馬上就把你的測試算進去 |
| G1 `api` | 免費——fdtdx API 表面、GPU 看得到、Meep 橋 ping |
| G3 `physics` | 免費——真空 flux 守恆,引擎層級的檢查 |
| G5 `crossengine` | 免費——介電平板上 fdtdx 對 Meep 對解析解 |
| G2 `gradcheck` Part C | 你寫 `gradcheck_case()`(設定、起始設計、`vg_fn`/`value_fn`);gate 負責訊號下限(signal floor:梯度低於峰值某比例的 voxel 不納入抽樣,常數是 `GRADCHECK_MIN_REL_GRAD`)、抽樣、Richardson 外推和 5% 的容差。Part A 與 Part B 是通用的,永遠都跑。 |
| G4 `reciprocity` | 你寫 `reciprocity_case()`,回傳兩個各自獨立歸一化的 dB 數字;gate 負責比較和 0.5 dB 的界線 |

兩個 case 都宣告在你模組的 `PROBLEM` 裡,而且兩個都是**用真的跑一次那道 gate**
來確認的:

```bash
uv run python scripts/00_check.py --only reciprocity --problem <your_problem>
uv run python scripts/00_check.py --only gradcheck   --problem <your_problem>
```

如果某道 gate 在你的 problem 上真的沒有東西可以檢查,那就**在程式碼裡把論證
寫出來**:

```python
reciprocity_case=Unsupported(
    "the measurement is p_bend / p_straight: two runs sharing one source and "
    "one normalization, so the normalization cancels in the ratio and there "
    "is nothing left to check")
```

(那是 `phc_bend` 真正的宣告。)runner 接著會印 `[n/a]`——或者在只有 gate 中
「量 problem」的那一半被宣告掉時印 `[part]`——並且把你的理由印在同一行。它不是
通過,也不是失敗,而且看起來不像其中任何一個。**你唯一不能做的是什麼都不說**:
那個欄位沒有預設值,所以沉默是一個 import error,而 runner 會把它變成
`[FAIL]`。理由要寫給「正在決定要不要相信你的數字」的那個人看,並且說清楚要
變成什麼樣,這道 gate 才會變得適用。

一個沒有引擎、沒有 GPU 卻兩道 gate 都拿得到的完整範例是
[`tests/fixture_problems/tmm_stack.py`](../tests/fixture_problems/tmm_stack.py)。

**自己加一道 gate 是「加一個檔案」,不是「去註冊」。** `gates/runner.discover()`
會把 `src/invdx/gates/` 底下所有名字以 `g` 開頭的模組 import 進來,依 `ORDER`
排序;runner 照那個順序執行,並在第一個失敗處停下來。`REQUIRES` 是寫給人看的
文件——**runner 不會去讀它**。

```python
"""Gate 6 — the slab problem's own physics anchor: toy-engine transmission
vs the analytic Airy curve, as a CURVE (conventions lesson 6), not at one
frequency.
"""

import numpy as np

from .runner import GateResult

NAME = "slab"
ORDER = 6
REQUIRES = ()          # 只是文件;runner 不會去讀它

TOL = 0.05


def run(cfg, args):
    from invdx.problems import slab

    scfg = slab.SlabConfig(n_freq=11, toy_steps=4000)
    T = np.array(slab.toy_transmission(scfg)["T"])
    Ta = slab.analytic_transmission(scfg)
    err = float(np.max(np.abs(T - Ta) / Ta))
    details = {"max_rel_err": err, "T_toy": T.tolist(),
               "T_analytic": Ta.tolist()}
    if err > TOL:
        return GateResult(NAME, "fail", {
            "reason": f"slab transmission deviates from the analytic Airy "
                      f"curve by {err:.1%} > {TOL:.0%}",
            **details})
    return GateResult(NAME, "ok", details)
```

把它存成 `src/invdx/gates/g6_slab.py`。

**確認它成了**——先確認找得到,再確認跑得動:

```bash
uv run python -c "
from invdx.gates import runner
print([(m.ORDER, m.NAME) for m in runner.discover()])
"
uv run python scripts/00_check.py --only slab
```

```
[(0, 'unit'), (1, 'api'), (2, 'gradcheck'), (3, 'physics'), (4, 'reciprocity'), (5, 'crossengine'), (6, 'slab')]
[ok]   G6 slab (18.61s)
```

有兩個慣例值得守住:一道失敗的 gate 的 `details["reason"]` 就是 runner 印在
那一行上的東西,所以要寫給沒讀過這道 gate 的人看;以及整個 `details` dict 都會
落進 `gates_report.json`,所以數字放那裡,不要放在 print 裡。

如果你的 problem 需要 GPU 或 Meep 環境,就在 `REQUIRES` 裡寫出來(給讀者看),
並且讓這道 gate 在前置條件不存在時直接丟例外——runner 會把那個變成 fail。
`gates/__init__.py` 解釋了為什麼「安靜地跳過」更糟:在一行摘要裡,**一道被跳過
的 gate 和一道通過的 gate 長得一模一樣**。

---

## 第 8 步:driver 腳本

腳本要薄。它們解析參數、組出 config、開一個 run 目錄、呼叫 problem 模組、把結果
存起來——**所有物理都住在 problem 模組裡**,而這正是測試和 gate 能重複使用它的
原因。

```python
#!/usr/bin/env python
"""Dielectric-slab transmission, stage by stage.

  python scripts/22_slab.py --stage eps      # look at the geometry
  python scripts/22_slab.py --stage measure  # T(f) vs the analytic anchor
"""

import os

import numpy as np

from invdx import runio
from invdx.cli import base_parser, apply_overrides, start_run
from invdx.problems import slab


def stage_eps(cfg, d):
    for layout in ("empty", "slab"):
        eps = slab.epsilon_grid(cfg, layout)
        np.save(os.path.join(d, f"eps_{layout}.npy"), eps)
        print(f"[{layout}] grid {eps.shape}, eps values {np.unique(eps)}")


def stage_measure(cfg, d):
    res = slab.toy_transmission(cfg)
    T = np.array(res["T"])
    Ta = slab.analytic_transmission(cfg)
    res["T_analytic"] = Ta.tolist()
    res["max_rel_err"] = float(np.max(np.abs(T - Ta) / Ta))
    runio.save_json(os.path.join(d, "transmission.json"), res)
    print("\n   f       T_toy    T_airy   rel err")
    for f, a, b in zip(cfg.freqs, T, Ta):
        print(f" {f:.4f}  {a:7.4f}  {b:7.4f}  {abs(a - b) / b:7.2%}")
    print(f"\n[anchor] max relative error vs Airy: {res['max_rel_err']:.2%}")

    from invdx.viz import plots
    plots.plot_transmission(
        [(cfg.freqs, T, "toy FDTD"), (cfg.freqs, Ta, "Airy (analytic)")],
        os.path.join(d, "transmission.png"),
        "slab transmission", ylabel="T")


def main():
    p = base_parser(__doc__)
    p.add_argument("--stage", default="eps", choices=("eps", "measure"))
    args = p.parse_args()
    cfg = apply_overrides(slab.SlabConfig(), args)
    d = start_run(cfg, args, "slab")
    {"eps": stage_eps, "measure": stage_measure}[args.stage](cfg, d)
    print(f"[done] {d}")


if __name__ == "__main__":
    main()
```

`start_run` 每次被呼叫都免費給你一個帶時間戳的 run 目錄,裡面有 `config.json`、
`cmdline.txt`、`env.txt` 和 `hardware.json`。

**確認它成了:**

```bash
uv run python scripts/22_slab.py --stage measure --set n_freq=7 \
    --set toy_steps=3000 --tag doc
```

```
[run] outputs -> runs/<timestamp>-slab-doc

   f       T_toy    T_airy   rel err
 0.4000   0.8305   0.8373    0.80%
 ...
[anchor] max relative error vs Airy: 0.80%
[done] runs/<timestamp>-slab-doc
```

### 不用自己寫繪圖程式就有的圖

`python -m invdx.viz <run-dir>` 會走過一個 run 目錄,把每一個它認得的檔名都畫出來。
**檔名照著寫,圖就是免費的:**

| 你寫出來的檔案 | `viz.render_run` 拿它做什麼 |
|---|---|
| `eps_*.npy` | 每個檔案畫一張介電常數分布圖 |
| `field_*.npz`,鍵有 `field`、`eps`,以及 `extent` 或 `extent_a`(`title` 選用) | 疊在介電常數分布上的穩態場圖 |
| `results.json` 裡含 `"history"` | 優化軌跡 |
| `results.json` 裡含 `"spectrum"` | 效率頻譜 |
| `design.npz`,鍵有 `eps` | 優化後設計的介電常數分布 |

```bash
uv run python -m invdx.viz runs/<dir>          # 要向量輸出就加 --pdf
```

有一個但書,因為在這裡猜錯的代價是一張標錯的圖:`gap.json` 和 `bend.json`
**不是**通用的掛勾。`render_run` 會用光子晶體的標籤,加上一段寫死的參考能隙
(`0.29–0.41`)去畫它們,因為它們屬於 `phc_bend`。**換成任何其他 problem,請
自己取檔名**,並像上面那支 driver 一樣直接呼叫 `plots.plot_transmission` /
`plots.plot_eps` / `plots.plot_field`。

---

## 第 9 步:逆向設計(需要才做)

**第 1 到第 6 步全過了才走到這一步。** 優化器會放大你的量測鏈所相信的任何東西。

[`optimize.py`](../src/invdx/optimize.py) 是真正與 problem 無關的那一塊:它既不
import fdtdx,也不 import 任何 problem 模組。它給你的是:在 `[0, 1]` 盒約束的
latent 向量上跑 Adam、`cfg.beta_schedule` 的退火 schedule、每次迭代的原子性
checkpoint(**斷點續跑**的那一種,見下面的命名陷阱)、續跑,以及依迭代次數 /
牆鐘預算 / 收斂三種條件停下來。

> **命名陷阱(先讀這個)**:這一節的 **checkpoint 有兩個互不相干的意思**,英文
> 原文同樣共用這個字:
> 1. **斷點續跑用的 checkpoint** —— `opt_state.npz`,每次迭代寫一次,讓
>    `resume` 接得回去。下面 `run_loop` 那一段講的都是這一個。
> 2. **梯度重算用的 checkpoint** —— `fdtdx.GradientConfig` 那個「checkpointed」,
>    反向傳播時只保留少數幾個時間點的狀態,其餘的場靠重算補回來,拿時間換記憶體。
>    下面第 2 點的可微分模擬場域講的是這一個。
>
> 兩者可以獨立調整,調錯邊的症狀完全不同:前者是續跑接不上,後者是記憶體爆掉
> (out of memory,log 裡寫 OOM)。

所有 problem 特有的東西都藏在一個 callable 後面:

```python
from invdx import optimize

state = optimize.run_loop(
    vg_fn,            # vg_fn(p, beta) -> (loss, grad),loss = -FOM
    p0,               # 起始的 latent 陣列
    cfg,              # beta_schedule 住在它身上
    n_iters=40,
    lr=0.02,
    run_dir=d,
    resume=False,
    time_budget_h=None,
)
```

這份契約,以及裡面容易做錯的部分:

- **`loss = -FOM`。** 這裡每一個 problem 都在**最大化**一個 FOM,所以
  `history.csv` 記的是 `CE = -loss`。**符號接反的 FOM 會朝著你的目標的反方向
  優化,而且把那個過程回報成「有進展」。**
- `vg_fn` 也可以改成回傳 `((loss, aux), grad)`,由 `aux` 裝真正的效率和一個懲罰
  項;history 的欄位定義在 `optimize.HISTORY_HEADER`。
- `n_iters` 是 beta schedule 的分母。**續跑時要保持不變**,否則退火行為會在你
  不知情的情況下整個變掉——
  `beta_for_iter` 的 docstring 和 `run_loop` 裡的續跑那條路都解釋了為什麼續跑時
  以 checkpoint 裡的 `beta` 為準。
- 迴圈每跑完一次迭代就原子性地寫一次 `opt_state.npz`(先寫 `.tmp.npz`,再
  `os.replace`),所以一次被中斷的 run 最多損失一次迭代。

**沒有通用工具、因此你一定得自己寫的部分:**可微分的模擬場域,以及被 trace 的
FOM。這個 repo 裡**沒有**與 problem 無關的 `Device` 工廠。`grating_coupler` 的
那幾份是一維專用的——例如 `design_device` 裝的是
`ConicFilter1D(radius_um=..., axis=0)` 再接 `fdtdx.TanhProjection`——所以請抄過去
改,不要 import:

1. `grating_coupler.design_device` —— 在設計視窗上放一個 `fdtdx.Device`,一個
   設計像素配一個 voxel,串上「濾波 → 投影」的參數鏈。
2. `grating_coupler.build_scene_design` —— 把量測用的模擬場域裡的光柵換成那個
   Device,再加上一個 checkpointed 的 `fdtdx.GradientConfig`。
3. `grating_coupler.te0_target_on_monitor` ＋ `grating_coupler.ce_from_arrays`
   —— 一個在 trace 之外只算一次的靜態目標,加上你那條量測鏈的 jnp 可微分版本,
   從跑完的結果上讀出來。
4. `grating_coupler.make_ce_value_and_grad` —— 把上面三件組成
   `jax.jit(jax.value_and_grad(loss))`。

**在你燒掉 GPU 小時之前先確認它成了:**用
[`richardson_fd.richardson_fd_check`](../src/invdx/richardson_fd.py) 對你的梯度
做有限差分,它是 `gates/g2_gradcheck.py` 和
`scripts/15_grating_coupler_optimize.py` 共用的那個核心。傳給它一個
`evaluate(sign, h) -> float` 的 closure,讓它擾動一個設計 voxel,它會從兩種步長
回報 `fd`、`rel_err` 和 `fd_consistency`。有兩條用昂貴代價學到、寫在
[`optimize.zh-TW.md`](optimize.zh-TW.md) 和
[`RETRACTIONS.zh-TW.md`](RETRACTIONS.zh-TW.md) 裡的規則:

- **只檢查梯度達到峰值一定比例的那些 voxel。** 低於那條訊號下限(signal floor)
  的地方,有限差分量到的是 float32 的捨入誤差,不是你的 adjoint。
- **不要用「調高容差」去回應一次 gradcheck 失敗。** 單一步長的 FD 可能是敗在
  **截斷**誤差上,而 adjoint 其實是對的——兩種步長的 Richardson 形式存在的理由
  就是把這兩件事分開。

然後把 [`optimize.zh-TW.md`](optimize.zh-TW.md) 整份讀完:Device 對方塊的等價性
檢查、為什麼把起始設計 rasterize 是物理,而不是換個格式而已,以及為什麼優化器自己印出來的
數字是一個排序訊號,永遠不是可以報出去的結果。

---

## 我怎麼知道自己做完了?

一個新的 problem 做完了,是指下面每一條都印出它該印的東西:

| # | 做完是指 | 指令 |
|---|---|---|
| 1 | config 能來回轉換,而且會拒絕打錯的鍵 | `uv run python scripts/<NN>_<your_problem>.py --set nonsense=1` → `unknown config key` |
| 2 | 幾何就是你想要的那個 | `--stage eps`,然後去看那個陣列或畫出來的圖 |
| 3 | 量測 run 和它的歸一化 run 只差在結構上 | 讀你自己的 `_run`:同樣的網格、同樣的步數、同樣的埠 |
| 4 | 結果和一個你沒有去湊的錨吻合 | 你的 `--stage measure` 會印出和錨的對照 |
| 5 | 幾何的不變量被測試釘住了 | `uv run python -m pytest tests/test_<your_problem>.py -q` |
| 6 | 整套測試還是全過 | `make check` |
| 7 | 那個錨是被自動強制的,不是靠人記得 | `uv run python scripts/00_check.py --only <your_problem>` → `[ok]` |
| 8 | 別人光靠 run 目錄就能重跑出你的結果 | `runs/<dir>/config.json` ＋ `cmdline.txt` 就能重現 |
| 9 | *(只有逆向設計要)* 在任何一次長 run 之前,梯度都對過有限差分 | 在正式設定下跑 `richardson_fd_check` |
| 10 | 兩道「量 problem」的 gate 都有答案,而且是你要的那個答案 | `--only gradcheck --problem <your_problem>` 與 `--only reciprocity --problem <your_problem>` → `[ok]`,或印著你寫的理由的 `[n/a]`/`[part]` |

**第 4 條和第 7 條要是缺了,你手上的是一次模擬,不是一個量測。**

---

## 誠實的地圖:哪些是通用的,哪些不是

寫下來是為了讓你**事先規劃**,而不是做到一半才發現。

| 模組 | 對一個新的 problem 而言 |
|---|---|
| `config.py`、`cli.py`、`runio.py` | 完全通用。不用改。 |
| `optimize.py` | 完全通用——不 import 任何引擎,也不 import 任何 problem 模組。 |
| `engines/conventions.py` | 通用規則;如果另一個 problem 也可能踩到,就把你的規則加在這裡,而不是加在你自己的 problem 模組裡。 |
| `engines/meep_bridge.py` | 通用。要加一個 Meep 任務,就是在 `engines/meep_worker.py` 加一個 `task_<name>(payload, jobdir)`,並在那個檔案的 `TASKS` dict 裡註冊;payload 必須是單純的 JSON,陣列要另外經由 `run_job(..., arrays={...})` 傳。來回檢查:`make smoke-meep`。 |
| `gates/g0`、`g1`、`g3`、`g5` | 與 problem 無關;你直接繼承。 |
| `gates/g2` Part C、`gates/g4` | 對「problem 提供的 case」做通用檢查。在你的 `PROBLEM` 裡宣告 `gradcheck_case` / `reciprocity_case` 就繼承得到,或是宣告 `Unsupported(reason)`,拿到一個**標示清楚的缺口**而不是一個安靜的缺口(第 7 步)。 |
| `viz/plots.py` | 由檔名驅動,大致上免費(見第 8 步),但 `gap.json` / `bend.json` 例外,它們帶著 `phc_bend` 的標籤。 |
| `report.py` | 從 `results.json` 讀幾個鍵(`peak`、`bandwidth_3db`、`linewidth`、`spectrum`、`corners`、`s11`)。吐一樣的鍵,Markdown 表格就會動;吐別的鍵,就自己寫。 |
| `export/gds.py` | 對一維二值剖面是通用的:`export_profile_gds(rho, grid_per_um=..., width_um=..., min_feature_um=..., out=...)`,外加一道最小線寬自檢。 |
| `export/handoff.py` | 有一部分是 `grating_coupler` 專用的。它會替任何一次 run 匯出設計向量、頻譜和 manifest,但碰到它不認得的 config 就會跳過介電常數的 raster,並在 manifest 的備註裡說出來。 |
| `datasets.py` | 目前只服務 `grating_coupler`。 |
| `fab/` | 通用:`ConicFilter1D` / `ConicFilter2D`、`min_feature_1d`、`erode_dilate_1d`、`softmin`、`tanh_projection`。 |

---

## 接下來讀哪裡

- [`phc-bend-walkthrough.zh-TW.md`](phc-bend-walkthrough.zh-TW.md) —— 同一批材料
  換成物理教學的講法:在兩個引擎上一步一條指令重現一個文獻基準。第 4 步的
  「把它錨住」如果是你最沒把握的那一段,就去讀它。
- [`optimize.zh-TW.md`](optimize.zh-TW.md) —— 第 9 步背後的全部細節。
- [`tolerance.zh-TW.md`](tolerance.zh-TW.md) —— 設計做出來之後,怎麼拿製程誤差
  去評它。
- [`env.zh-TW.md`](env.zh-TW.md) —— 環境的兩層分工,萬一這一頁有什麼東西
  import 不起來就去讀它。
