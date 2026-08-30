> [English](README.md) · **繁體中文**

# 文件索引

每一份文件配一句話,說它管的是什麼;專案本身的總覽在最上層的
[`README.zh-TW.md`](../README.zh-TW.md)。
**語言:** [English](../README.md) · 繁體中文。

下面列出的每一份,結尾都標了它有哪些語言。所謂雙語是**一對檔案**——英文
`X.md` 旁邊放一份繁體中文 `X.zh-TW.md`,兩份的第一行互相連到對方。
(〈匯出〉那一節列的是指令而不是文件,所以不帶語言標記。)

`make bilingual` 用機械的方式查這些對子——程式碼區塊、連結、標題層級、
互指行——然後執行 [`glossary.zh-TW.md`](glossary.zh-TW.md) 上的裁定:那份表
引到的每一個檔名與節名都必須解得開,它禁掉的寫法不准在中文樹的任何一頁活著,
而它綁給某個英文詞的中文寫法,不准出現在那個英文詞缺席的地方。每一道檢查都會
印出它究竟看了多少東西,所以分母縮水看得見。它**不查散文**;散文只有冷讀
查得到。

這一頁和最上層的 `README.md` 自己也被同一條規則綁著,而且底下點名的文件裡
還有一份沒做到,所以這裡寫的是**規則本身**,不是對整棵樹現況的描述:
**精確的數字不准寫進沒有人會重算的散文裡。**單元測試有幾支、雙語有幾對、
某個模組幾行——這些都曾經被人手抄一次,幾週內就錯了,而過期的數字讀起來和
新鮮的一模一樣。所以這兩頁上的數字,要嘛出自算它的那道指令(`--problem` 的
說明行會列出登記過的 problem;`make bilingual` 會印出它找到的對子,並在樓地板
掉下去時失敗;`make runs` 會列出 run 目錄),要嘛就換回它原本想講的那句定性
宣稱。留在散文裡的數字只有一種:讀者不必離開這一頁就能證偽它。「六道關卡」
留了下來,因為六道在句子底下的表裡一列一道全部點名,工作流程圖裡又出現一次
——多出第七道的那一刻,那兩個字會當場錯得看得出來。「178 支測試」沒有留下來,
因為這一頁上沒有任何東西能反駁它。

已知的例外是 [`dependencies.zh-TW.md`](dependencies.zh-TW.md):它還留著三個手抄的數字
——Python 環境的套件總數、native(層)環境的套件總數,以及 `nvidia-*` wheel
的數量。沒有任何東西會重算它們:每一個都只差一道對 `uv.lock` 或
`spack/env/spack.lock` 下的 `grep -c`,在那條線接起來之前,它們的新鮮度就
只等於上一個回去重推的人。把那一頁收進這條規則底下是**已知的待辦**,不是
這段話沒察覺的疏漏。

## 術語

- [`glossary.zh-TW.md`](glossary.zh-TW.md) —— 中文文件裁定了哪些詞、為什麼:
  同一個概念長出兩個中文名的那些、中文的直覺譯法早就被別的意思佔走的那些
  碰撞、刻意留英文不譯的那些,以及首次出現就必須就地定義的符號。它是一次次
  冷讀長出來的,不是靠事先設想出來的。它**只放裁定**;「某某檔現在寫成什麼」
  這類對現況的宣稱一律不寫在上面,改由 `make bilingual` 每次重算——因為
  這一頁自己就是那些宣稱會被改掉的原因。
  **語言:** [English](glossary.md) · 繁體中文。

## 教學

- [`phc-bend-walkthrough.zh-TW.md`](phc-bend-walkthrough.zh-TW.md) —— 親手做,
  一步一道指令,在兩支引擎上重現光子晶體 90° 彎那個文獻經典基準。
  **語言:** [English](phc-bend-walkthrough.md) · 繁體中文。
- [`../tutorials/01-jax-port/`](../tutorials/01-jax-port/) —— 第一課:把一支
  二維 FDTD 移植到 JAX,骨架刻意留了洞給你自己填。
  **語言:** [English](../tutorials/01-jax-port/README.md) ·
  [繁體中文](../tutorials/01-jax-port/README.zh-TW.md)——課程頁和它的
  `RESULTS` 參考輸出兩份都有。
- [`../tutorials/02-first-adjoint/`](../tutorials/02-first-adjoint/) —— 第二課:
  你的第一個 adjoint 梯度,拿有限差分去對答案。
  **語言:** [English](../tutorials/02-first-adjoint/README.md) ·
  [繁體中文](../tutorials/02-first-adjoint/README.zh-TW.md)——課程頁和它的
  `RESULTS` 參考輸出兩份都有。

## 環境與重現

- [`env.zh-TW.md`](env.zh-TW.md) —— uv 與 spack 這兩層怎麼切、架構圖、從乾淨
  clone 重現的步驟,以及給新手的兩層各自入門:一個新相依該進哪一層、
  `uv.lock` 和 `spack.lock` 各自釘死了什麼、離線/無網路那條路以及關於它
  真正被量到的東西、漂移檢查,還有兩層各自踩過的坑。
  **語言:** [English](env.md) · 繁體中文。
- `bash scripts/bootstrap.sh` —— L1 層(uv、JAX、fdtdx):裝起來、拿 GPU
  `driver` 版本當一道關卡擋(這裡的 `driver` 是 L0 的 GPU 驅動程式,不是把
  優化迴圈串起來跑的那支主程式)、再 import 進來驗它真的裝對了。
  `bash spack/bootstrap.sh` 是它在 L2(Meep)的對應。兩支都是冪等的;
  `make env-drift` 查已提交的 lockfile 是不是還對得上已提交的意圖。
- [`dependencies.zh-TW.md`](dependencies.zh-TW.md) —— 這個工具箱站在什麼東西
  上面:每個套件由誰維護、授權加起來是什麼(包含 GPL 引擎是怎麼被隔離開的),
  以及少掉其中一個會壞掉什麼。
  **語言:** [English](dependencies.md) · 繁體中文。

## 怎麼做

- [`new-problem.zh-TW.md`](new-problem.zh-TW.md) —— 把你自己的元件加進工具箱:
  一個 problem 模組非提供不可的東西有哪些、該從哪一個檔案抄起、在付錢跑模擬
  之前先怎麼把幾何看一遍,以及那些**會安靜地給你錯答案而不是拋例外**的慣例
  契約。兩道真的去量一個具體元件的關卡(G2 Part C 與 G4),只要宣告一個
  `ProblemSpec` 就繼承過來——不要的話得寫下來,理由由那道關卡自己印出;忘了
  表態會是一個 import error,而不是安靜地少掉一塊覆蓋率。
  **語言:** [English](new-problem.md) · 繁體中文。

## 方法筆記

- [`optimize.zh-TW.md`](optimize.zh-TW.md) —— 逆向設計迴圈:可微分的 Device
  路徑、FOM、Richardson gradcheck、斷點續跑用的 `checkpoint`(`opt_state.npz`
  那一種,不是梯度重算用的 `num_checkpoints`),以及怎麼在 Slurm 上跑。
  **語言:** [English](optimize.md) · 繁體中文。
- [`tolerance.zh-TW.md`](tolerance.zh-TW.md) —— 為容差而設計的方法筆記,以及
  回報時的慣例(敏感度圖、corner 評估)。
  **語言:** [English](tolerance.md) · 繁體中文。

## 匯出

- `python -m invdx.export.handoff <run-dir>`(或 `make handoff RUN=…`)——
  一個工具中立的包裹:介電常數網格、頻譜、設計向量、manifest。
- `python -m invdx.export.gds --design <run-dir>` —— GDS-II 佈局,附一道最小
  線寬自檢;同時寫出 `<out>.gds.fingerprint.json`,那是幾何契約在匯出端的
  那一半。
- `invdx.export.contract` —— 對匯出的多邊形取指紋,並把轉檔工具產出的結果
  讀回來比對,好讓一次交付是被查過的,而不是被相信的。

## 誠實紀錄

- [`journal.zh-TW.md`](journal.zh-TW.md) —— 只增不改的工作日誌;每一個報出來
  的數字都註明它出自哪一次 run、哪一個 commit、哪一份報告檔。
  **語言:** [English](journal.md) · 繁體中文。
- [`RETRACTIONS.zh-TW.md`](RETRACTIONS.zh-TW.md) —— 這個專案發表過、後來發現
  是錯的那些結論:原地更正,並留一條指回這裡的線索,而不是安靜地改掉。
  **語言:** [English](RETRACTIONS.md) · 繁體中文。
