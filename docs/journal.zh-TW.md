> [English](journal.md) · **繁體中文**

[← back to docs index](README.md)

# 工作日誌(只追加)

照時間順序記下「實際上發生了什麼」。每個數字都附上它的來源(log 檔、
報告路徑,或承載那個結果的原始碼/測試檔)。條目只往後追加、不回頭改寫;
結論後來被推翻的話,寫進 `RETRACTIONS.zh-TW.md`,而不是回來把這裡的歷史
改掉。

唯一的例外發生在 2026-11-05:為了讓這份日誌只描述工具本身,各條目裡關於量測時
所用範例元件的細節都刪掉了。沒改過的版本留在 git 歷史裡。

**引用有兩種,先分清楚哪一種點得開:**

| 引用長這樣 | 點得開嗎 | 它在這裡的作用 |
|---|---|---|
| `runs/…`、`spack/env/install.log` | ✗ 本機的 run 產物,不在版控裡(見 `.gitignore`) | 標明某個數字出自哪一次 run,不是給讀者點的連結 |
| `src/`、`scripts/`、`tests/`、`spack/` 底下的路徑 | ✓ 就在這個 repo 裡 | 讀者可以直接打開來對照的引用 |

> **讀這份日誌的四個提醒**
> 1. 同一個日期出現好幾筆是正常的——那是同一天的不同工作線,不是重複記錄。
> 2. 沒過門檻的東西照樣寫進來,而且**不回頭把門檻調鬆**。這份日誌的用途是
>    留下證據,不是展示成果。
> 3. 術語一律沿用程式與報告裡的英文原詞(方便你直接 grep 原始碼),
>    第一次出現時給一句中文解釋。
> 4. 上面那張引用對照表、這個提醒方塊、下面的術語速查,以及正文裡幾個標了
>    括號的補充說明,都是中文版專屬的導讀,英文版沒有對應段落。正文的事實、
>    數字與來源兩邊一致。

**術語速查**(全篇沿用這些寫法):

| 詞 | 一句話 |
|---|---|
| gate / G0–G5 | 驗收關卡:過就 pass、不過就把流程擋下來的檢查。`make gates` 一次跑完整套 |
| V3 | 另一套編號,屬於這次里程碑的驗證項目,**和 G0–G5 這套 gate 不是同一套** |
| gradcheck | 用有限差分(FD, finite difference)去驗 adjoint 梯度算得對不對的檢查 |
| adjoint | 伴隨法:一次正向加一次反向就拿到所有設計參數的梯度(和線性代數的「伴隨矩陣」是兩回事) |
| FOM | figure of merit,被優化的那個目標值。不寫成「品質因數」——那個中文在光學裡已經被 Q factor 佔走了 |
| h / h-scan | h 是有限差分的步長;h-scan 就是掃一串不同的 h 看殘差怎麼變 |
| binarization gap | 連續設計值與硬二值化之後的表現落差;binarization schedule 沒跑完(β 還小)時落差本來就會大 |
| schedule | 這裡專指 β 隨迭代升高的預定表,不是 Slurm 那種工作排程 |
| checkpoint | optimizer 的狀態存檔(含 β 與參數),用來續跑或收尾。**本檔的 checkpoint 一律是這個意思**,不是反向傳播拿時間換記憶體的那種 |
| signal floor | 訊號下限:梯度低於峰值 5% 的 voxel 不納入 gradcheck 抽樣 |
| driver | 把整條優化迴圈串起來跑的主程式(**不是**裝置驅動程式;GPU driver 那個意思本檔沒有用到) |
| fast / vanilla | 兩條前向模擬路徑:`vanilla` 是 fdtdx 原本的 `run_fdtd`,`fast` 是本 repo 的加速前向迴圈,兩者的場逐位元相同 |
| hot | 計時方式:第一次跑(含 JIT 編譯)不計入,量的是之後的穩態速度 |
| view | spack 把整條相依鏈 hardlink 成一棵可以直接拿來用的目錄樹 |

> **TODO(待裁決,尚未有結論)**:上面這張術語速查是中文側獨有的增補,而
> 「該回填英文側,還是宣告為本地化補強」**還沒有人裁決**。判準與兩個選項見
> `glossary.zh-TW.md`〈六、中文獨有的增補:回填英文,還是宣告為本地化補強〉;
> 依那一節的規定,裁定必須寫在這裡(增補自己旁邊),不寫進詞彙表。在裁定寫下
> 來之前,不要把這張表當成已經宣告為本地化補強——上面第 4 點只是描述現況,
> 不是裁定。

---

## 2026-08-17 — 環境可重現性(L1 uv、L2 spack)

**L1:uv 接管 Python/GPU 這一層**

- `pyproject.toml` 釘住 `jax[cuda12]==0.11.0` + `fdtdx==0.6.2`;`uv.lock`
  凍結 148 個套件。既有的 conda env 原封不動(留作退路)。
- 驗收:G0–G5 全綠(G5 是 `[ok]`,不是被 skip 掉);benchmark 拿
  `runs/benchmark_fast_v6.json` 當基準對過——fast 量到 1250.832、基準是
  1250.799 Mcell-steps/s,vanilla 量到 636.748、基準是 638.033,兩者都是暖機
  之後(hot)的數字;`peak_mem_gb` 兩邊同為 6.015903;`max|dE|=max|dH|=0.0`,
  也就是兩條路徑的場必須恰好相同,不容許任何數值漂移。來源:
  `runs/bench_uv_verify.json`。

**L2:meep 改由 spack 從原始碼編**

- spack 釘了兩層:工具本身 `v1.2.0` + packages repo `v2026.06.0`
  (`spack/env/spack.yaml`)。106 個套件的相依鏈,`reuse: false`,用系統的
  gcc 11.4,hardlink view 落在 `spack/env/.spack-env/view`
  (= `meep_bridge.py` 的預設路徑)。
- 編譯事故:第一次安裝就卡在 meep 本身——SWIG 4.4.1 產生的 binding 撞上
  `structmember.h` 的 `READONLY` 巨集與 `meep.hpp` 的 enum 同名(SWIG ≥4.1
  的已知衝突,上游在 1.29 之後修掉了)。解法:把 swig 釘成 `swig @=4.0.2`
  (`=` 是 spack 的「精確版本」語法;寫成 `@4.0.2` 不加 `=` 會匹配到
  swig-fortran 那個 fork——這是個坑)。第二次安裝:meep@1.29.0,2m51s 編完。
  來源:`spack/env/install.log`。
- 驗收:完全不啟用 spack 環境(不跑 activate),直接用裸的 `view/bin/python`
  執行 `import meep` 就拿得到 1.29.0;`mpirun -np 2` 看得到 2 個 rank
  (MPI 的平行行程);G5 跨引擎一致性
  `T_analytic=0.73978, T_fdtdx=0.74412, T_meep=0.74230`——fdtdx 對 meep 差
  0.24%(容差 10%)。乾淨演練:重新 clone 一份 + `spack install`
  (1.3s,只建 view)+ `make smoke-meep` → 1.29.0。來源:
  `runs/20260817-022058-gates/gates_report.json`。
- 已知的量測瑕疵(記的是量測失效,不是結果):有一次完整的 `make gates` 在
  G2 Part C 就 OOM(out of memory,記憶體爆掉)了,原因是當時另一個開發用的
  run 同時佔著兩張卡的 GPU 記憶體(JAX 會預先把記憶體吃下來)。**這不是
  spack 的問題**;那一次的 gates 還沒有重跑,待辦是要在乾淨的 GPU 上一次
  只跑一項地重跑一遍。來源:`runs/20260817-021820-gates/gates_report.json`。

## 2026-08-17 — 第一支正式運轉(production)用的逆向設計 driver

- 新的可微路徑:在設計視窗上放一個 `fdtdx.Device`,一個設計像素對一個 voxel,
  取代舊的區塊幾何路線——那條路線會先把設計二值化,從來就不可微。FOM 用 jnp
  重寫了一份與量測鏈的模態重疊積分等價的可微分版本;V3 一致性檢查(V 開頭的編號
  是這次里程碑的驗證項目,和 G0–G5 這套 gate 不是同一套)在同一個二值設計上比對
  「可微 FOM」與區塊幾何那條鏈,兩條路線吻合到 |Δ| = 2.5e-6 dB(門檻 0.05)。
- G2 多出 Part C(在一條真實的設計路徑上對 3 個 voxel 做 FD):相對誤差
  0.0008% / 0.177% / 0.067%,對的是 5% 容差,並且配一個 signal floor 抽樣器
  (500 個 voxel 裡有 278 個在峰值梯度 5% 以上,才有資格被抽——對 tanh 已經
  飽和的 voxel 做 FD,拿到的是 float32 的相消(cancellation)雜訊,所以**沒有**
  因此放寬容差)。六個 gate 都在 HEAD 這個 commit 上驗過,而且分別出自彼此獨立
  的 run(`runs/20260817-032448-gates` 是 G0–G4 在乾淨 GPU 上一次只跑一項;G5 是
  修完之後重跑,99.73 秒)。
- 實作過程中掉出兩個發現:(1) 起始設計原本的 rasterize 用的是「格點中心落在
  哪就算哪」的規則,在 20 nm 網格上讓特徵寬度起伏 ±3.5%,光是這一項就讓 FOM
  變動一個數量級以上;rasterize 現在改成照 fdtdx 擺 block 的方式把邊界取整。
  (2) `VAR= cmd` 這種寫法會把環境變數設成空字串(「設成空的」和「沒設」在 shell
  裡是兩回事),而 bridge 把它當成一條路徑用——`meep_bridge.py` 現在遇到「有設
  但是空的」會退回預設 view。
- 迴圈 smoke 測試(0.15 ps、3 個迭代,其中一個是續跑):FOM 跨過續跑的邊界
  仍然單調變好,binarization gap 在起始 β 下小到可以忽略。
- 在 CPU/GPU 有其他負載互搶的情況下量到的每次迭代耗時有記下來,但不可以當成
  效能數字引用(量測紀律)。

## 2026-08-17(夜間)— Slurm 開跑;一次完整尺寸的 run 被自己的安全 gate 擋下來

- Slurm 這條路在本地叢集上驗過了:`gres` 會正確地把 `CUDA_VISIBLE_DEVICES`
  0/1 分給同時在跑的 job,sbatch 腳本不用改就能從 job ID 推出自己的 run
  目錄。一次 sbatch smoke 測試,加上一次「迭代跑到一半 scancel 再 requeue」的
  演練,兩個都通過——`history.csv` 的迭代 0,1,2,3 跨過砍掉/續跑的邊界仍然
  連續,沒有重複的列。
- 第一次正式的完整尺寸 run 被跑前的 gradcheck 擋下來:voxel 213,相對誤差
  6.29%,對的是 5% 容差,而且在固定 seed 下可重現,所以就記成
  `gradcheck_failed`,而不是默默重試一次。另一組設定的 gradcheck 過了
  (1.93%),整晚跑完。
- 根因來自在 GPU 上做的 h-scan:**是 FD 的截斷誤差,不是 adjoint 有缺陷**
  ——殘差隨 h³ 縮放(h 每減半就 ×0.126),用同一批 run 做 Richardson 外推的
  結果與 adjoint 吻合到 0.0106%,float32 的雜訊底比觀察到的殘差低 200 倍,
  而且出問題的那些 voxel 並不是低訊號的。跑得久(0.8 ps 對 0.15 ps)會把
  FOM 在設計空間裡的曲率放大,這就是為什麼 smoke 規模的檢查會過。先前把
  這個失效模式歸因於「float32 相消誤差」的說法**已經撤回**——見
  `RETRACTIONS.zh-TW.md`。修法:改成兩個 h 的 Richardson gradcheck,外加一個
  FD 自洽指標(優化 driver 裡的 `gradcheck()`,`src/invdx/gates/g2_gradcheck.py`
  Part C 有一份鏡像);容差與 signal floor 都沒有動。

**L2 第二階段:改用專案自己維護的 spack package repo,把 meep 升到 1.34.0**
(這是同一天的另一條工作線)

- `spack/spack_repo/invdx/` 放的是上游 meep recipe(spack 的 package 定義檔)
  的**完整複本**,不是繼承來的 subclass——因為 spack 的 constraint 在繼承
  之下只能收緊、不能放寬。三處改動:`version("1.34.0", ...)`、python 版本
  條件化(`@:1.31` 配 `@:3.11`,`@1.32:` 配 `3.11:3.13`)、`@1.32:` 起要求
  `py-numpy@2:`。
- sha256 的發現:同一個 v1.34.0 tag 底下有兩份官方產物。GitHub release 的
  dist tarball(`3c9284…60bc6`,conda-forge 用的那份)少了 `python/numpy.i`,
  會在 `No rule to make target 'numpy.i'` 掛掉;git tag 的 archive
  (`1fa6dd…78ea4`,GitHub 依 tag 自動打包的原始碼)配 autoreconf 才建得
  起來。兩個 hash 都直接對 github.com 驗過。recipe 用的是 git tag 的
  archive。
- SWIG 實驗:把當初為了 1.29 的 READONLY 衝突而下的 `@=4.0.2` 釘版拿掉,
  meep 1.34.0 用 swig 4.4.1 乾淨編過,3m16s——確認上游確實修了。現在整條鏈
  是:python 3.13.13、numpy 2.4.6(與 conda baseline 同一個世代)。
- 驗收:裸 view 直接 import 到 1.34.0;2 個 MPI rank;`make smoke-meep` 走
  預設路徑回報 1.34.0;G5 從頭重跑兩次(一次在 GPU 1,再加一次獨立重跑)
  ——`T_meep=0.7423000529144463`,在這個 case 上與 1.29 的結果逐位元相同;
  重跑 `concretize --force`(spack 把抽象需求重新解成一組確定版本)之後,
  lock 仍然解出同一份結果。來源:
  `runs/20260817-025817-gates/gates_report.json`、`spack/env/install.log`。

## 2026-08-17 — Bug:續跑時 beta 是重算的,不是從 checkpoint 讀

從兩次刻意停掉的 run 回收設計時(兩者在磁碟上真正的 beta 都是 64),浮出一個
續跑的 bug:`run_loop` 進迴圈之前的 `state.beta` 是
`beta_for_iter(cfg, iteration, n_iters)` 現算出來的——用的是「這次續跑呼叫
傳進來的 `n_iters`」,而不是從 `opt_state.npz` 讀。

把一個被砍掉的 run 收尾,最自然的做法就是 `--resume <dir> --iters <iters_done>`
(不多跑任何一次迭代,只是想把最終設計寫出來)。問題在於那個旗標同時扮演兩個
角色:它既是「還要跑幾次迭代」,也是 schedule 的分母。分母一縮小,同一個
iteration 就落到更後面的階段,於是無聲地跳一級——兩次 run 都重算出 beta=128,
而真值是 beta=64。這本來會把一個比實際更銳利的 TanhProjection 固化進
`design_rho.npy` / `design_rho_cont.npy` 裡;實際上並沒有發生,因為當時就繞開
了:另外寫一支用完即丟的腳本(未 commit),直接從磁碟讀 `state.beta`,不走
`--resume`。

修法:`run_loop` 現在續跑時改用 checkpoint 自己的 `beta` 欄位去種
`state.beta`,只有全新開始時才用 `beta_for_iter`;schedule 仍然照常推進,
因為迴圈內部每一次迭代都還是會依當下的 `it`/`n_iters` 重算 `beta`。driver 另外
多了 `--finalize-only`:載入 checkpoint、以零次 optimizer 迭代重跑收尾段
(`design_rho*.npy`、`results.json`)——就是那支拋棄式腳本的功能,升格成一個
正式旗標,升格之後把拋棄式腳本刪掉。拿真實的 run 驗過:`--finalize-only` 重現
出來的 `design_rho.npy` 與拋棄式腳本產物逐位元組相同(sha256 吻合)。回歸測試在
`tests/` 底下。

## 2026-11-05 — 重寫 README,並拿程式碼本身核對一遍

在一台 4 核的 Intel Xeon 雲端 VM 上(只有 CPU,沒有 Meep),把不需要 GPU 的
部分重跑了一次。

- `pytest tests/`(也就是 G0 包起來的那一套):320 個通過、1 個跳過(那個測試
  需要一個真正的 run 目錄),4 核上跑 9 分鐘。
- toy 引擎的 adjoint 腳本沒通過它自己的 gradcheck:有一個像素的相對誤差是
  2.28e-5,超過 1e-5 的門檻,而所有 adjoint 數值都和記錄下來的輸出一模一樣。
  掃一遍 h 之後,問題落在 float64 的捨入誤差上:h = 1e-4 時是 1e-6 到 2e-5,
  h = 1e-3 時是 1.1e-7 到 1.9e-7,h = 1e-2 時是 1.6e-5(換成截斷誤差主導)。
  h 現在是 1e-3;重跑通過(最差 1.92e-7),而且優化那幾行印出來的每一位都和
  記錄下來的輸出相同。
- `docs/dependencies.zh-TW.md` 把「優化塞得進一張 GPU」歸功於時間可逆的反向
  傳播。正式運轉(production)用的路徑其實是 checkpoint(梯度重算用的那種),
  而可逆模式在這些模擬場域(scene)上反而更差,這件事先前就量過了。
- README 和文件現在只描述工具本身;圍繞著特定範例元件寫的那幾頁都刪掉了。

README 引用的數字各自出自哪裡:G5 與 G2 Part C,見上面 2026-08-17 的條目;G4 是
正向對反向差 0.076 dB(關卡門檻 0.5 dB),在 commit `1b326e5` 那次出處欄位的
改動前後逐位元相同;快速正向路徑與單卡容量,出自
`src/invdx/engines/fdtdx_perf.py` 的模組 docstring(在 Quadro RTX 6000 上量的);
adjoint 記憶體模型,見 `RETRACTIONS.zh-TW.md` 2026-08-22 那一條。
