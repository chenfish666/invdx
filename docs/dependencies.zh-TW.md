> [English](dependencies.md) · **繁體中文**

[← back to docs index](README.md)

# 相依盤點:這套工具站在什麼東西上面

這頁回答四個問題:這套工具**站在什麼東西上面**、每一塊**是誰在維護**、
所有授權**加起來是什麼**、以及**少了其中任何一個會壞掉什麼**。

兩個數字先框住規模:Python 環境解出 **148 個套件**(`uv.lock`),
native 環境解出 **122 個套件**(`spack/env/spack.lock`)。這裡面幾乎沒有
一個是本專案自己挑的——下面的表只收兩種:程式碼真的 `import` 到的,
以及雖然是被別人帶進來、但在授權或資安上有後果值得點名的。

**先認幾個詞**(英文原詞就是你在 `uv.lock`、工具訊息和上游文件裡會看到的那個字):

| 詞 | 這是什麼 |
|---|---|
| process(不譯) | 作業系統裡一個獨立執行的程式實例;兩個 process 之間不共用記憶體。本頁授權那一節整段論證就掛在這條界線上,所以這個字讀錯,整頁都會讀錯。中文「行程」會被讀成旅遊行程,因此保留英文 |
| pin(精確鎖版) | 用 `==` 把版本鎖到單一一版,不給浮動空間。本專案只 pin 兩個套件 |
| transitive(間接相依) | 不是你宣告的,是別的套件把它拉進環境裡的 |
| vendor / vendored | 把上游的一段程式碼複製進自己 repo 自行維護,不再走套件安裝 |
| copyleft | 「改了/散布了就得比照開放」那一類授權(GPL、LGPL、MPL);中文常譯成「傳染性授權」,見下方陷阱方塊 |
| wheel | Python 的預編譯安裝包。裝得快,但裝進來的是**二進位成品**,不是原始碼 |
| native(原生編譯層) | 不從 wheel 裝、直接在本機從原始碼編譯出來的那一層(Meep、MPI、HDF5) |
| ABI | 二進位層面的介面約定。對不上的話,版本號看起來相容也還是會爆 |
| advisory(資安通報) | 已公開的漏洞紀錄。本頁的通報數來自 OSV 掃描 |

> **中文讀者的用詞陷阱(先讀這個)**:中文圈談 GPL 慣用「病毒式授權」「傳染」
> 這組詞,而這組詞會讓你把「共存在同一個環境裡」誤讀成「已經被感染」。
> 真正的判準從來不是遠近,而是兩件很具體的事:**程式碼有沒有進到同一個
> process、你有沒有把它散布出去**。本專案兩個 copyleft 引擎剛好落在這條線的
> 正反面——Meep(GPL)只透過另起一個 process 呼叫,界線是**架構**做出來的;
> tidy3d(LGPL)是 `import` 進同一個 process 的,界線在這裡不是架構撐出來的,
> 是因為 LGPL 這張授權本身就管得鬆。另外,MPL 在中文討論裡常被順手歸進
> 「寬鬆授權」,它不是:它是檔案層級的弱 copyleft。

> **這一頁的寫法說明(這是翻譯上的取捨,不是專案的技術主張)**:授權名稱一律
> 保留英文原名(`BSD-3-Clause`、`LGPL-2.1-or-later` ……),因為那才是能拿去查、
> 能跟 lockfile 欄位逐字比對的字串;中文意譯名在任何資料庫裡都沒有對應物。

## 兩層環境,兩種壞法

本專案刻意跑在兩個從不共用同一個 process 的環境上。

**Python 層**(`uv.lock`,Python 3.12)裝的是可微分的設計路徑:JAX、FDTDX
引擎、優化器、匯出工具。它從 wheel 安裝,幾秒鐘裝完,壞的時候就是 wheel
這套安裝方式會壞的那幾種壞法:相依解不開、JAX 和它的 CUDA plugin 之間 ABI
對不上、某個間接相依落後上游太久。

**Native 層**(`spack/env/spack.lock`,Python 3.13)裝的是 Meep 和它那套
MPI/HDF5/GSL——也就是獨立的交叉驗證引擎。它從原始碼編譯,要跑幾個小時,
壞的時候是原始碼編譯會壞的那種壞法:編譯器或 MPI 版本不合、少一個系統
標頭檔(header)、或者更難纏的——**編譯成功了,但算出來的數值有細微差異**。

兩邊的 Python 直譯器是真的各自編譯出來的兩個 build(3.12.14 與 3.13.13),
這才是切分的真正原因,不是為了風格好看:JAX/CUDA 和一個 Meep build 沒辦法
被解進同一個環境。它們之間**只靠檔案溝通**——`engines/meep_bridge.py` 把
`job.json` 加上 `.npy` 陣列寫進一個 job 目錄,起一支
`mpirun -np N <spack-view>/bin/python meep_worker.py <jobdir>`,擋著等它跑完,
再把 `result.json` 讀回來。兩個世界之間共用的介面,就只有 NumPy 的 `.npy`
這一個格式。

## 「維護者」這一欄怎麼讀

要判斷「這東西能不能依賴」,關鍵不是它紅不紅,而是**壞掉的時候誰扛**:

- **基金會 / 公司** —— 有機構經費、有數名支薪維護者、有棄用(deprecation)
  週期和資安流程。要壞會先公告。
- **學術實驗室** —— 有經費的研究團隊做的研究軟體。通常品質不錯、背後有論文,
  但撐著它的就那幾個人,而經費會斷、人也會走。API 穩定性只能算
  盡力而為(best-effort)。
- **單一維護者** —— 某一個人的專案,再優秀也一樣。風險不在程式碼品質,
  而在於**沒有第二個人**。

本專案照這條規矩走:機構級的套件可以放心依賴;學術與單一維護者的套件一律
精確 pin 住,並且擺在一層可替換的介面後面。

## Python 層 —— 自己宣告、程式直接 import 的

| 套件 | 鎖定版本 | 用在哪 | 誰在維護 | 授權 |
|---|---|---|---|---|
| `numpy` | 2.4.6 | 到處都是。陣列型別本身,也是 Meep bridge 跨環境交換資料的格式 | 基金會(NumFOCUS) | BSD-3-Clause(內含 vendored 的 0BSD / MIT / Zlib / CC0 片段) |
| `scipy` | 1.18.0 | 只有一處 import:`fab/measure.py` 裡用 `scipy.ndimage` 做膨脹/侵蝕(dilation/erosion)這兩個形態學運算,量最小線寬。**從不出現在可微分路徑上** | 基金會(NumFOCUS) | BSD-3-Clause |
| `autograd` | 1.9.1 | numpy 側的自動微分:`fab/filters_np.py`、`modes.py`,而**最要緊的是** `gates/g2_gradcheck.py`——它在那裡當「獨立的參考梯度」,用來對照檢查 JAX 路徑 | 個別志工(源自某學術實驗室;創始作者已經不再貢獻程式碼) | MIT |
| `gdstk` | 1.0.1 | GDSII 佈局的寫出與讀回驗證:`export/gds.py`、`export/contract.py` | 單一維護者 | BSL-1.0(Boost;不是 BUSL) |
| `pyevtk` | 1.7.0 | 只用一個函式 `imageToVTK`,在 `export/vtk.py` 裡產 ParaView 輸出 | 單一維護者,小型協作組織 | MIT |
| `jax` | 0.11.0(`==` pin 死) | GPU 上的可微分核心。30 個檔案 import 它:`optimize.py`、`fab/filters_jax.py`、`toy/fdtd2d_jax.py`,以及整個 `engines/fdtdx_*` 家族 | 公司(Google) | Apache-2.0 |
| `fdtdx` | 0.6.2(`==` pin 死) | GPU FDTD 引擎,19 個檔案 import。選它的理由是 `reversible_fdtd`——時間可逆的反向傳播,記憶體對時間步數是 O(1) | 學術實驗室(登在 JOSS,經同行審查) | MIT |
| `optax` | 0.2.8 | 四行:`optimize.py` 和一支 toy 腳本裡的 `optax.adam` 與 `optax.apply_updates` | 公司(Google DeepMind) | Apache-2.0 |
| `equinox` | 0.13.8 | **`pyproject.toml` 裡任何地方都沒宣告它。** `engines/fdtdx_checkpoint_buffers.py` 直接 import `equinox.internal` 來用 `buffers=` 這個關鍵字引數(kwarg)。它純粹是被 fdtdx 當作間接相依帶進來的 | 單一維護者 | Apache-2.0 |
| `matplotlib` | 3.11.1 | 宣告在 `[dev]` 底下,卻被主線進入點用到:`viz/plots.py`(`make viz`)和容差報告(`make tolerance`) | 基金會(NumFOCUS) | 以 PSF 為基礎的寬鬆授權(非 copyleft) |
| `pytest` | 9.1.1 | 只有測試套件用 | 不是基金會本身,但屬同一圈生態;多位維護者 | MIT |

整個環境只有兩個東西被精確 pin 死:`jax==0.11.0` 和 `fdtdx==0.6.2`。
fdtdx 那個 pin 不是保守起見,是**拆掉就會壞**:`engines/fdtdx_fixes.py` 裡有一份
vendored 的子類別,用來修 0.6.2 的高斯源(Gaussian source)一個軸序錯誤,
而那份修補只對 0.6.2 有效。其餘所有套件在 `pyproject.toml` 裡都是浮動的,
靠 `uv.lock` 定住。

有兩個宣告上的問題,與其永遠寫在文件裡,不如去修掉。第一,`equinox` 被直接
import 卻從未宣告,等於整個 build 建立在「fdtdx 會繼續把它拉進來」這個假設上;
更糟的是那個 import 伸進了 `equinox.internal`——一個**沒有任何相容性承諾的
私有 API**,所以真正比較可能發生的不是套件消失,而是某次不起眼的小版本升級
悄悄把它改掉。第二,`matplotlib` 待在 `[dev]` 裡,但有兩個 `make` 目標需要它,
所以一個只裝正式運轉(production)那組相依的環境,會缺一個主線相依。

## Python 層 —— 被別人帶進來的

| 套件 | 鎖定版本 | 為什麼會在這裡 | 誰在維護 | 授權 |
|---|---|---|---|---|
| `jaxlib`、`jax-cuda12-plugin`、`jax-cuda12-pjrt` | 0.11.0 | JAX 的 XLA 後端與 CUDA 執行 plugin。從不被直接 import,但少了它們 JAX 就只剩 CPU。版本必須跟 `jax` 完全一致 | 公司(Google) | Apache-2.0 |
| 13 個 `nvidia-*-cu12` | cuBLAS 12.9.2.10、cuDNN 9.24.0.43、NCCL 2.31.2,以及另外 10 個 | 由 `jax[cuda12]` 拉進來,jaxlib 動態載入。專案沒有任何一行程式碼碰它們 | 公司(NVIDIA) | **`LicenseRef-NVIDIA-Proprietary`** —— 不是開源 |
| `tidy3d` | 2.12.0 | 專案程式碼裡零次提及,但它是 fdtdx 的硬相依,fdtdx 用它做模態求解(mode solving)。只要 `import fdtdx`,它就被載進同一個 process | 公司(Flexcompute) | **LGPL-2.1-or-later** |
| `pillow` | 11.3.0 | 經由 matplotlib、imageio、moviepy 進來。只有在 matplotlib 寫 PNG 時才會被碰到 | 社群組織(多維護者) | MIT-CMU |
| `certifi` | 2026.7.22 | 間接相依,走 tidy3d 那條鏈 | 有公司支持 | **MPL-2.0** |
| `tqdm` | 4.70.0 | 間接相依,走 tidy3d 那條鏈 | 社群 | **MPL-2.0 AND MIT** |

tidy3d 這個相依值得白話講清楚,因為**從專案自己的原始碼裡完全看不見它**。
它是一家商用雲端 FDTD 廠商的用戶端函式庫(client SDK),而它會把自己整套
用戶端相依一起帶進環境:AWS SDK(`boto3`/`botocore`)、HTTP 與認證函式庫、
一整套 MCP server 程式,還有接到作業系統金鑰儲存(keyring)的綁定。實際做過
載入測試:`import fdtdx` 的時候,
那些網路與認證模組**一個都沒有被 import**,跟著進來的只有資料層那幾個
(`dask`、`xarray`、`pandas`、`shapely`、`h5py`)。所以沒有偷偷連回原廠
(phone-home)這回事。但它們確實被裝進來了,裝進來就是攻擊面,
而一個離線的 HPC 專案根本用不到它們。

資安上有一項是開著的。把 148 個鎖定套件全部拿去 OSV 掃,只中一個:
`pillow` 11.3.0,帶 36 筆通報。這些全部是影像**解碼**的漏洞:例如解 JPEG2000
的時候,一個個分塊 tile 的緩衝區會一直堆積不釋放,最後把記憶體吃光
(8.2.0 引入、12.3.0 修好)。而本專案
從頭到尾只透過 matplotlib **寫出** PNG,從不去解析來路不明的影像檔,
所以真實暴露程度很低。

而且這件事**在這個 repo 裡修不掉**。這點要講精確,因為那個看起來理所當然
的解法真的不管用:`uv lock --upgrade-package pillow` 跑得完、回報 148 個
套件已解析,然後把 `pillow` 留在 11.3.0——沒有錯誤、沒有警告。要**指名
要求那個修好的版本**,理由才會浮出來:

    uv lock --upgrade-package "pillow==12.3.0"
    # ... moviepy 2.2.1 depends on pillow<12.0,>=9.2.0
    # ... fdtdx>=0.6.2 depends on moviepy>=2.1.1
    # ... your project's requirements are unsatisfiable

也就是說,這個版本上限是**結構性的**——`fdtdx` → `moviepy` → `pillow<12.0`
——它要解開,得等 `moviepy` 放寬它的約束、或 `fdtdx` 不再依賴它,
而不是這個 repo 裡改什麼東西。native 層不在這條鏈上,它裝的已經是
`py-pillow` 12.2.0,這也就是兩層版本會愈拉愈開的原因。

## Native 層 —— Meep 環境

由 Spack 從原始碼建置。它的 lockfile(把每個套件的確切版本寫死的那份檔案)
鎖了兩層:一層是套件規格本身,另一層是把 Spack 的 package repository 鎖在
tag `v2026.06.0`。少了第二層,實際上根本談不上可重現。

| 元件 | 版本 | 角色 |
|---|---|---|
| `meep` | 1.34.0 | 交叉驗證用的 FDTD 引擎。用本 repo 自帶的 package recipe(spack 的 package 定義檔)建置,variant(建置選項)為 `+python +mpi +hdf5 +libctl +harminv +mpb +gsl +openmp` |
| `mpich` | 5.0.1 | MPI。選它是為了對齊參考用的 conda Meep build,減少跨引擎的變因 |
| `hdf5` | 1.14.6 | Meep 的場輸出格式 |
| `libctl`、`harminv`、`mpb` | 4.5.1、1.4.2、1.11.1 | Meep 自家的支援函式庫——控制語言、諧波反演、模態求解器 |
| `fftw`、`gsl`、`openblas` | 3.3.11、2.8、0.3.33 | Meep 底下的數值計算 |
| `py-numpy`、`py-scipy`、`py-mpi4py`、`py-matplotlib` | 2.4.6、1.17.1、4.1.1、3.11.0 | worker 那一側的 Python 環境。`runio.py` 用 `mpi4py`,輸出會看 rank(MPI 平行 process 的編號)決定怎麼做 |

`spack.lock` 完全沒有記錄任何套件的授權欄位,所以下面這些 native 授權是
**上游專案自己的宣告**,不是從安裝好的產物上讀出來的。Meep 是
GPL-2.0-or-later;GSL 是 GPL-3.0-or-later;FFTW 是 GPL-2.0-or-later。
這是整個系統裡授權包袱最重的一塊,同時也是**隔離得最徹底**的一塊——見下一節。

還有一點:兩層對同樣的函式庫**刻意**鎖了不同版本(這裡 `py-scipy` 是 1.17.1,
Python 層是 1.18.0)。它們是兩個各自獨立的環境;兩邊版本剛好一致是巧合,
不是約束。

## 授權加起來是什麼

專案本身是 MIT。相依樹裡沒有任何東西改變這一點,但有三件事必須明說,
不能當成理所當然:

**GPL 在這裡是真的存在,而它是靠 process 邊界隔離的。** Meep 和它底下那些
GPL 數值函式庫,從來沒有被連結進本專案、也沒有被 vendored 進來。碰到它們的
唯一方式是**另起一支獨立的程式**——不同的 Python 直譯器、不同的環境、
一棵完全不相交的相依樹——然後透過一個 job 目錄交換 JSON 和 `.npy` 檔案。
沒有任何 GPL 程式碼被 import 進「同時裝著專案程式碼」的那個 process,
而專案也沒有散布其中任何一份。這座「另起一支 process」的橋常被說成是
「為了繞過一個解不開的環境衝突而做的權宜之計」;它同時也正是讓 GPL 義務
傳不過來的東西。

**Python 樹裡的 copyleft 套件有三個,不是一個。** `tidy3d` 是
LGPL-2.1-or-later(套件自己宣告的是 "v2 or later";只寫 `LGPL-2.1` 會低估它)。
`certifi` 是 MPL-2.0,`tqdm` 是 `MPL-2.0 AND MIT`——MPL 是檔案層級的弱
copyleft,不是寬鬆授權。這三個都是原樣使用、未經修改,也都沒有被再散布,
所以義務停留在那些檔案上,碰不到專案程式碼。不過跟 Meep 不一樣的是:
tidy3d 是**跑在同一個 process 裡的**,所以這裡的界線不是架構分出來的,
是 LGPL 這張授權本身就管得鬆。最上層 README 的 "Engine licenses" 一節記的就是同樣這三個。

**唯一一塊真正閉源的是 NVIDIA 的。** 那 13 個 CUDA runtime wheel 是
`LicenseRef-NVIDIA-Proprietary`。這是整個系統裡唯一一處讓「完全開源、
完全可重現」這句話站不住的地方,而它值得精確點名,正是因為從原始碼角度
看過去會被誤導:NCCL 的**原始碼**是 BSD-3-Clause,但**實際裝進來的那個**
`nvidia-nccl-cu12` wheel 宣告的是專有條款。本頁記錄的一律是**安裝產物自己
宣告了什麼**。

資安掃描還有一個但書:OSV 對專有二進位套件的覆蓋本來就不完整,所以
NVIDIA 那些 wheel 「沒有通報」的意思是「查不到紀錄」,不是「沒有漏洞」。
那一塊的權威是原廠自己的資安公告,而本頁沒有查過那些公告。

## 少了其中一個會怎樣

**專案直接結束。** `numpy` 和 `jax` 與其說是相依套件,不如說是這個專案的
地基——numpy 還額外身兼兩個環境之間唯一的共通格式。`jaxlib` 和那些 CUDA
wheel 則是按版本焊死在 JAX 上的。

**最貴的那一個。** `fdtdx` 是整個系統裡代價最高的相依。時間可逆的 FDTD
反向傳播——記憶體對時間步數是 O(1) 而不是 O(T)——正是三維優化能塞進
單張 23 GB GPU 的唯一原因。每個替代方案都各有一個很具體的地方輸掉:
Meep 的伴隨法(adjoint)求解器綁在 CPU 上,慢一到兩個數量級;Tidy3D 是
商用的,而且跑在雲端;repo 內的 `toy/fdtd2d_jax.py` 是二維教學用實作,
離能正式運轉(production)還差得很遠。針對這一項有一個刻意做的緩解措施:
`export/handoff.py` 會輸出一包不綁工具的資料(介電常數網格、設計向量、
耦合效率頻譜,以及一份 manifest——說明這包裡有什麼的清單檔),讓**設計結果**
不被鎖死在產生它的那個引擎上。

**換掉很便宜。** `scipy` 貢獻的是兩個只拿來做量測的形態學函式;拿 3×3
結構元素自己手寫大約一小時的工,而且**不可能改變任何物理結果**。
`pyevtk` 貢獻一個函式,對應的是一個已公開的檔案格式——30 到 50 行,
或直接換 `meshio`。`optax` 貢獻 Adam 和 `apply_updates`,大約 20 行 JAX;
它留著的唯一理由是 fdtdx 反正也會把它裝進來。

**換得掉,但要付代價。** 換掉 `gdstk` 意味著重寫 GDS 匯出和它的讀回檢查,
不過 GDSII 是標準格式,所以沒有任何資料會被困住;`gdsfactory` 或 KLayout
API 都能接手。`autograd` 沒了,壞掉的不是功能,是方法:它存在的意義,就是
在 gate G2(本專案六道驗收關卡 G0–G5 裡,管梯度算得對不對的那一道)當一份
**非 JAX** 的第二意見來檢查梯度;若改用 JAX 自己來做,這個檢查賴以成立的
「跨框架獨立性」就整個塌掉了。現有的 Richardson 有限差分(以外推法求數值
梯度)那條路可以頂替,但精度較低。

**這一項要問的是另一個問題。** `equinox` 不太可能消失;就算 `buffers=`
這個優化沒了,程式碼也會退回 fdtdx 未使用 buffer 的路徑——多吃記憶體,
結果一樣。真正實際會發生的失效是:私有的 `equinox.internal` API 在一次
例行升級底下悄悄變動。那是**監控問題,不是替換問題**,而第一步是把這個
套件連同一個相容的版本範圍正式宣告出來,不要再靠 fdtdx 順手供應。

## 這份盤點已知的缺口

明說出來而不藏起來,好讓下一輪知道從哪裡開始:

- 建置層沒有涵蓋。`setuptools>=68` 是 `[build-system]` 的需求,
  在 `uv.lock` 裡完全找不到,因為 uv 不鎖 build backend(負責把原始碼
  打包成 wheel 的那個工具)。
- native 的授權是上游宣告,不是從安裝好的檔案裡驗證出來的。
- 維運用的那套外掛工具在版本控制之外,也不屬於對外散布的專案,
  所以它自己的 import(`pyyaml`、`nvidia-ml-py`、`psutil`)是**刻意**
  排除在這裡的。它們都被 `try/except` 包住並附有明確的安裝指示,
  而且三個裡有兩個根本不在環境裡——那是本機工具鏈的事,不是專案相依。
- 套件數、授權、資安通報都是**當下時點的快照**。它們隨時可以從
  `uv.lock` 和 `spack/env/spack.lock` 重新導出,那兩份才是權威紀錄;
  這一頁是對它們的一次閱讀,不是它們的替代品。
