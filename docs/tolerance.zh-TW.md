> [English](tolerance.md) · **繁體中文**

[← back to docs index](README.md)

# 容差設計:方法筆記

這份檔案要解決的問題:逆向設計優化時只對著**一個**標稱幾何(nominal
geometry),但製程實際做出來的是一群彼此有偏差的幾何。本專案附的範例問題是
一個光柵耦合器(grating coupler),假設 193 nm DUV SOI 平台。針對這一類製程
的模擬研究指出:逆向設計出來的元件即使滿足**硬性**的最小線寬約束,只要還沒
做過考慮微影的修正,良率仍然可能是 Yield_90% = 0%(OptoSynthesizer,
arXiv:2604.15493,Table 1 —— 模擬,沒有實測 fab 資料)。所以最小線寬規則
**值得拿來回報,但不值得信任**。這份筆記做兩件事:把那批文獻的作法對應到
本專案已經有的機制上,以及先把回報慣例釘死,讓別人每次引用「容差」這個數字
時,意思都一樣。

## 怎麼跑

```bash
make tolerance RUN=runs/<coupler-opt-dir>                 # sensitivity map + corner evaluation
make tolerance RUN=runs/<dir> LAMS=1.27,1.35,9         # plus corner CE spectra
make runs                                              # which run dirs qualify
```

第一條跑敏感度圖加 corner 評估。第二條會多算每個 corner 的 CE(coupling
efficiency,耦合效率,本專案要最大化的那個量)頻譜,`LAMS` 三個數字依序是
起始波長、結束波長、取樣數(波長單位 um)。第三條列出哪些 run 目錄可以餵給
verify / tolerance / handoff:`runs/` 底下什麼都有(gate 的 run、benchmark、
被砍掉的優化、完成的設計),`make runs` 會把帶有設計向量或結果檔的目錄挑
出來,並標出每個目錄各自能拿來做什麼。

> **命名陷阱**:這裡的 corner 是**製程角**,指製程參數走到極端時的那幾組
> 設定,不是幾何上的轉角(`phc-bend-walkthrough.zh-TW.md` 裡那個 90° 彎才是
> 轉角)。中文文獻常見的「角點」則是影像處理的 corner detection,更不是這個
> 意思。所以本檔一律直接寫 corner,不翻。

輸出寫進 `<run-dir>/tolerance/`,目前是三個檔:`sensitivity.csv`、
`sensitivity.png`、`corners.csv`;來源 run 目錄全程唯讀,不會被改到。實作在
`scripts/16_tolerance_report.py`,所有旗標以
`python scripts/16_tolerance_report.py --help` 的輸出為準。

更正一件事:這份檔案的舊版本把下面這些工具寫成「計畫中的工作」,但它們當時
其實早就寫完了。讀者於是很合理地以為這個功能還不存在——正好和事實相反。

## 本專案已經有的東西

- `src/invdx/fab/transforms.py` —— conic filter(圓錐核密度濾波,半徑 =
  `cfg.min_feature`)接上 tanh projection(tanh 投影,把連續的密度值往 0/1
  推)。projection 的閾值 eta 就是控制 erosion/dilation 的那個參數:
  `eta_i = 0.5`(標稱)、`eta_e`(eroded,對應 over-etch/過蝕)、`eta_d`
  (dilated,對應 under-etch/蝕刻不足),三個都已經是 `src/invdx/config.py`
  裡的欄位。這組 eta 就是形態學 erosion/dilation 的標準寫法,下面 corner
  評估重新 rasterize 時用的也正是這三個值。
- 貫穿整個模擬場域(scene)的 adjoint 梯度(伴隨法:跑一次正向、一次反向,
  就拿到全部設計參數的梯度);梯度正不正確由 G2 這道 gate 把關——G2 用中央
  有限差分驗算 adjoint 梯度,`make gates` 會依序跑到它。所以敏感度分析幾乎
  不用額外成本。

## 報告會算什麼

兩個步驟都實作在 `scripts/16_tolerance_report.py`:

1. **敏感度圖(sensitivity map)** —— 在最終設計上求 `∂CE/∂rho`(rho 是設計
   密度場,也就是優化真正在動的那組連續變數,最後存成 `design_rho.npy`),
   再化簡成逐齒的線寬敏感度(齒 tooth 指光柵的週期性線條,逐齒就是逐條線
   去看)。成本:一次反向傳播(backward pass)。產出:一張圖加一份 CSV,把
   「哪些齒在線寬漂移下最會拖垮 CE」排出名次。理由:光罩到晶圓
   (mask-to-wafer)的誤差不是均勻的,表現好壞由敏感區決定
   (arXiv:2604.15493 §3.1.2)。
2. **corner 評估** —— 把固定不動的那個設計,拿三組 projection corner
   (`eta_i` / `eta_e` / `eta_d`)重新 rasterize(用這三組 eta 重畫成 0/1
   幾何),逐個 corner 回報 CE 與 3 dB 頻寬。不重新優化;這一步量的是標稱
   設計有多脆弱。

這一步只是評估一個固定不變的設計,不是 robust optimization(穩健優化)。要把
這些 corner 搬**進**目標函數裡——也就是對三個 projection corner 這個集合
(ensemble)取最壞值(worst-case)或 softmin——那是另外一套公式,本專案沒有
實作。

這也是為什麼開頭說最小線寬「不值得信任」:把 `cfg.min_feature` 拿去當 conic
filter 的半徑,只有在三個密度場(three-field:eroded、nominal、dilated 各算
一次)的穩健公式底下,才**保證**做出來的線不會小於那個尺寸。本專案只優化
標稱的那一個密度場,所以那個半徑是啟發式的,什麼都保證不了——最小尺寸要用
量的,不能用假設的。

## 回報慣例

- 良率指標:`Yield_90% = Pr(CE_corner >= 0.9 * CE_nominal)`,在抽樣到的那組
  製程變異上計算(預設就是上面那三個 corner)。這裡的 0.9 倍是拿線性的 CE
  去比,不是拿 dB 值:腳本先把 `CE_dB` 換算回線性 CE 再比較。門檻在這份方法
  筆記裡就先定死,免得看到結果之後才回頭挑數字。
- corner 表格的欄位固定是:`corner, CE_dB, bw_3db_nm, ridge_lam_um`,依序是
  corner 名稱、該 corner 的 CE(dB)、3 dB 頻寬(nm)、CE 峰值所在的波長
  (um)。最後那個欄位名裡的 ridge 指頻譜上的峰形,不是脊狀波導(ridge
  waveguide)。
- 預設只跑單一波長(中心波長 `cfg.lam_c`),此時 `bw_3db_nm` 與 `ridge_lam_um`
  兩欄會留空——那是預期行為,不是壞掉;要填滿它們就加 `LAMS=LO,HI,N`(依序是
  起始波長、結束波長、取樣數,波長單位 um),讓它跑一次真正的頻譜掃描。
- 引用 `Yield_90%` 時把 n 一起寫出來。中文的「良率」兩個字很容易被讀成晶圓
  統計,但預設的 corner 評估 n=3,那是脆弱度檢查,不是統計良率估計(腳本
  自己印出來的那一行也是這樣寫的)。
- 本專案的所有結果都是模擬。要寫「模擬顯示」,絕不要寫「實驗顯示」。(引用
  arXiv:2604.15493 時同樣適用——那篇的 fab 結果是 digital twin(數位分身)
  模擬,不是量到的晶圓。)

## 參考文獻

- OptoSynthesizer: end-to-end physical design automation for yield-optimized
  inverse-designed EPICs. arXiv:2604.15493 (2026)。針對逆向設計 EPIC
  (電子-光子整合電路)的良率導向端到端實體設計自動化。本文開頭那個
  Yield_90% = 0% 出自這裡,「用敏感度導引修正」這個論點也是。
- BOSON⁻¹: variation-aware photonic inverse design, DATE 2025。變異感知的
  光子逆向設計;要把穩健公式放進優化迴圈裡就參考它——本專案沒有實作這條路。
- PRISM: photonics-informed inverse lithography, arXiv:2602.15762。光罩層級
  的修正(inverse lithography,逆向微影);本專案沒有光罩流程,所以不在範圍
  內,引用它是為了「敏感度不均勻」這個論點。
