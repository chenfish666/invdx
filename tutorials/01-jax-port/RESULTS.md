# 第一課產出(2026-07-26 完成,主線由 Claude 代寫)

物理主線的填答在 [src/invdx/toy/fdtd2d_jax.py](../../src/invdx/toy/fdtd2d_jax.py)
(空格 A/B/C 的參考解);本檔記錄驗收的實際輸出,親手練習時可對照。

## 驗收輸出(scripts/08_toy_jax_lesson1.py --gpu)

```
[case] 光子晶體 bulk 能隙量測,格點 110^2、2000 步(小尺寸,秒級)
[diff] max|dE| = 8.882e-16, max|dH| = 8.882e-16 (場量級 4.140e-01)
[time] numpy 0.27s | jax 首跑 1.28s(含編譯)| jax 再跑 0.28s
[PASS] 兩個引擎逐位元同物理 —— 你的第一個 JAX FDTD 成立。
[gpu] jax 預設裝置:gpu (NVIDIA RTX 6000 Ada Generation)
[gpu] 同一份程式碼零修改跑在 gpu:首跑 0.30s、再跑 0.37s
```

## 數字怎麼讀

- **8.9e-16**:float64 的機器精度(≈2.2e-16)乘上幾步的累積。這不是
  「兩個引擎很接近」,是「兩個引擎是同一個物理」——每一步浮點運算
  都相同,只是執行方式不同(直譯 numpy vs XLA 編譯)。
- **jax 首跑 1.28s vs 再跑 0.28s**:差額 ≈1s 是描圖+編譯,只付一次。
  同形狀的後續呼叫直接用編譯產物。
- **GPU 0.37s 沒有比 CPU 0.28s 快**:這個問題太小(110²),塞不滿
  GPU;搬資料的開銷吃掉了算力優勢。GPU 的回報要在大網格(pvgc 3D
  是 62.7M cells)或 float32 時才展現——「小問題別上 GPU」本身就是
  一課。
- **測試轉正**:tests/test_toy_jax.py 從此在每次 `make gates` 盯著
  兩引擎等價,JAX 版再也不能悄悄偏離 numpy 版。

## 這一課埋的伏筆

E 場更新裡那個 `/ eps[1:-1, 1:-1]` 現在活在一個 XLA 可微分程式裡。
第二課(tutorials/02)就對它下手:`jax.grad(透射率)(eps)` 一次拿到
「每個格點的材料怎麼影響輸出」——伴隨法,自動的。
