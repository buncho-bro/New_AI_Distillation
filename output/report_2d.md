# Concept Core: 自己組織化ポテンシャル場 AI アーキテクチャ

## 実験結果サマリー

| シナリオ | 検証内容 | 結果 |
|:---|:---|:---:|
| **A** Y字型の谷の形成 | 犬(A)→四足歩行(C)、猫(B)→四足歩行(C)の浸食で合流谷が自律形成されるか | **PASS** |
| **B** ビー玉推論テスト | チャネル内に投下したビー玉が勾配に沿ってC（共通概念）へ収束するか | **PASS** |
| **C** 風化ノイズ耐性 | 単発ノイズが風化処理（500ステップ）で消滅するか（残存 < 10%） | **PASS** |

---

## シナリオ A: Y字型の谷の形成（抽象化の検証）

60サイクルの浸食（各サイクルで A→C と B→C の刺激 + 2回の風化）を実行した結果:

- **C（四足歩行）の深度**: −2.45 （最深部）
- **A→Cチャネル中間点**: −1.54
- **B→Cチャネル中間点**: −1.54
- **背景（未接触）**: 0.00

> [!TIP]
> C が背景より **2.45 深い** — 二つの水流が合流する地点に最深の渓谷が自律形成され、「四足歩行」という共通概念のアトラクターが形成されたことを示しています。

````carousel
![Y字型の谷（等高線図）](C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/scenario_a_contour.png)
<!-- slide -->
![Y字型の谷（3Dサーフェス）](C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/scenario_a_3d.png)
<!-- slide -->
![断面プロファイル](C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/cross_section.png)
````

---

## シナリオ B: ビー玉推論テスト（アトラクター引き込みの検証）

学習済み地形上の異なる3箇所にビー玉を投下:

| ビー玉 | 開始位置 | 最終位置 | Cまでの距離 | ステップ数 |
|:---|:---|:---|:---:|:---:|
| 1 (A-channel内) | (30, 60) | (50.0, 28.0) | **3.0** | 233 |
| 2 (合流channel内) | (50, 40) | (50.0, 28.0) | **3.0** | 97 |
| 3 (B-channel内) | (70, 60) | (50.0, 28.0) | **3.0** | 233 |

> [!IMPORTANT]
> 全てのビー玉が **C = (50, 25) の近傍（距離 ≈ 3.0）** で静止しました。異なるチャネルからの入力が同一のアトラクター（共通概念）に収束することが確認されました。

![ビー玉の軌跡](C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/scenario_b_marble.png)

---

## シナリオ C: 風化テスト（ノイズ耐性の検証）

- **ノイズ地点 D**: (66.4, 46.3) に1回のみ刺激を注入
- **刺激直後の深度**: −0.0483
- **500ステップ風化後の深度**: −0.0021
- **残存率**: **4.4%** （< 10% 閾値 → PASS）

> [!NOTE]
> 単発のノイズによる浅い窪みは風化処理により自然に平坦化されます。一方、60回繰り返し浸食された主要チャネル（深度 −2.45）は風化に対して頑健に維持されます。

![風化前後の比較](C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/scenario_c_weathering.png)

---

## 物理パラメータ（調整可能）

| パラメータ | 記号 | デフォルト値 | 説明 |
|:---|:---:|:---:|:---|
| 浸食率（学習率） | $\eta$ | 0.05 | 1回の刺激通過あたりの最大掘削深度 |
| 風化率（忘却係数） | $\lambda$ | 0.005 | 1ステップあたりの平滑化強度 |
| 摩擦係数（速度減衰） | $\gamma$ | 0.85 | ビー玉の速度に乗じる減衰率 |
| 重力加速度 | — | 5.0 | 勾配力のスケーリング |
| 浸食半径 | — | 3.0 px | チャネル断面のガウシアン幅 |
| 時間刻み | $\Delta t$ | 0.5 | ビー玉シミュレーションの積分ステップ |
| 収束閾値 | — | 0.001 | この速度以下でビー玉が静止と判定 |

---

## プロジェクト構成

```
concept_core/
├── concept_core.py      # コアモジュール（ConceptCore, PhysicsParams）
├── run_experiments.py    # 実験シナリオ A/B/C の自動実行と可視化
└── output/               # 生成画像・アニメーション
    ├── scenario_a_contour.png
    ├── scenario_a_3d.png
    ├── scenario_a_evolution.gif
    ├── scenario_b_marble.png
    ├── scenario_c_weathering.png
    └── cross_section.png
```

## 使い方

```python
from concept_core import ConceptCore, PhysicsParams

# パラメータのカスタマイズ
params = PhysicsParams(erosion_rate=0.1, weathering_rate=0.01)
core = ConceptCore(params)

# 学習（浸食）
core.erode([(20, 80), (50, 25)])   # 軌跡に沿って谷を形成

# 忘却（風化）
core.weathering()                   # 1ステップの平滑化

# 推論（ビー玉）
final_pos, trajectory = core.roll_marble((30, 60))
print(f"収束先: {final_pos}")
```

実行コマンド:
```bash
python -X utf8 run_experiments.py
```
