> [English](README.md) · **繁體中文**

# 文件索引

每一份文件配一句話,說它管的是什麼;專案本身的總覽在最上層的
[`README.zh-TW.md`](../README.zh-TW.md)。
**語言:** [English](../README.md) · 繁體中文。

下面列出的每一份旁邊都有一份繁體中文版(`X.zh-TW.md`),兩份的第一行互相連到
對方。〈匯出〉那一節列的是指令而不是文件。

## 教學

- [`../tutorials/01-jax-port/`](../tutorials/01-jax-port/) —— 把一支
  二維 FDTD 移植到 JAX,骨架刻意留了洞給你自己填。
  **語言:** [English](../tutorials/01-jax-port/README.md) ·
  [繁體中文](../tutorials/01-jax-port/README.zh-TW.md)——課程頁和它的
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

## 怎麼做

- [`new-problem.zh-TW.md`](new-problem.zh-TW.md) —— 加入你自己的 problem:
  一個 problem 模組要宣告什麼、在付錢跑模擬之前先怎麼把幾何看一遍,以及那些
  **會安靜地給你錯答案而不是拋例外**的慣例。兩道真的去量一個具體 problem 的
  關卡(G2 Part C 與 G4),只要宣告一個 `ProblemSpec` 就繼承過來——不要的話得
  寫下來,理由由那道關卡自己印出;忘了表態會是一個 import error,而不是安靜地
  少掉一塊覆蓋率。
  **語言:** [English](new-problem.md) · 繁體中文。

## 匯出

- `python -m invdx.export.handoff <run-dir>`(或 `make handoff RUN=…`)——
  一個工具中立的包裹:介電常數網格、頻譜、設計向量、manifest。
- `python -m invdx.export.gds --design <run-dir>` —— GDS-II 佈局,附一道最小
  線寬自檢;同時寫出 `<out>.gds.fingerprint.json`,那是幾何契約在匯出端的
  那一半。
- `invdx.export.contract` —— 對匯出的多邊形取指紋,並把轉檔工具產出的結果
  讀回來比對,好讓一次交付是被查過的,而不是被相信的。
