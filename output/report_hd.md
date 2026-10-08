# High-Dimensional Concept Core (64-D) 実験レポート

PyTorchを用いて、2Dのグリッドベースだった「概念核」のトイモデルを64次元ベクトル空間へとスケールアップしました。高次元でのメモリ効率と計算を両立させるため、ポテンシャル場を動的ガウシアンカーネルバンク（Parametric Kernel Density）として定式化しています。

## 1. 高次元アーキテクチャの設計

- **ポテンシャル場（パラメトリック表現）**: 
  $$H(\mathbf{x}) = - \sum_{i=1}^{M} w_i \cdot \exp\left( -\frac{\Vert{}\mathbf{x} - \mathbf{c}_i\Vert{}^2}{2\sigma^2} \right)$$
  空間全体を保持する代わりに、浸食された谷の中心 $\mathbf{c}_i \in \mathbb{R}^{64}$ と深さの重み $w_i$ のみを動的に保存・管理（メモリバンク化）。
- **推論（自動微分による物理運動）**: 
  PyTorchの Autograd（`backward()`）を用いて任意の入力 $\mathbf{x}$ に対する勾配 $\nabla H(\mathbf{x})$ を直接計算。慣性と摩擦を伴うSGDライクな運動方程式でビー玉（粒子）を転がします。
- **過剰加速の防止 (Terminal Velocity)**:
  深い谷底付近では重み $w_i$ の増大に伴い勾配ベクトルが非常に大きくなるため、速度上限（Terminal Velocity / 勾配クリッピング）を導入してオーバーシュートを防いでいます。

---

## 2. 実験結果サマリー (64次元空間)

| シナリオ | 検証内容 | 結果 | 備考 |
|:---|:---|:---:|:---|
| **A** Y字谷の形成 | ランダムな64Dベクトル A, B から原点 C への軌跡を50回浸食 | **PASS** | 19個のカーネルに圧縮。<br>Cの深度: −64.0 （背景は 0.0） |
| **B** ビー玉推論 | (A+B)/2 の中間点から推論開始。C（原点）に収束するか | **PASS** | $L_2$距離 `4.0` からスタートし、最終的にCからの距離 `0.8` 未満の谷底で安定。 |
| **C** 風化ノイズ耐性| 単発ノイズを注入後、300ステップの風化を実行 | **PASS** | 閾値以下に減衰したカーネルは自動的にメモリからプルーニング（削除）され、0個に。 |

---

## 3. PCA射影による可視化

64次元空間内のアトラクター中心、浸食軌跡、推論軌跡のすべてを含む点をPCA（主成分分析）にかけ、上位2軸の平面へ射影しました。

> [!NOTE]
> A, B, C の3点が張る超平面上に主要なダイナミクスが集中しているため、**PCAの分散説明率は 99.5%** となり、次元削減後も元の構造（Y字型の谷）が美しく保持されています。
> グラフ右側の収束曲線からは、慣性（モメンタム）の影響で深い谷の底（C）付近で微細に振動しながら静止する物理的な挙動が確認できます。

![64-D PCA 射影と収束曲線](C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/high_dim_result.png)

---

## 4. 使い方とモジュール構成

プロジェクトは [`concept_core/`](file:///C:/Users/lop3l/.gemini/antigravity/scratch/concept_core) に配置されています。

- [`high_dim_concept_core.py`](file:///C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/high_dim_concept_core.py) : PyTorchによる `HighDimConceptCore` クラス本体
- [`run_high_dim_experiments.py`](file:///C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/run_high_dim_experiments.py) : 実験スクリプト（シナリオ A〜C の実行と PCA プロット）

```python
import torch
from high_dim_concept_core import HighDimConceptCore

# 64次元モデルの初期化
core = HighDimConceptCore()

# 浸食（学習）
trajectory = torch.stack([start_vec, target_vec])
core.erode(trajectory)

# 風化（忘却とプルーニング）
pruned_count = core.weather()

# 推論（ビー玉）: 自動微分で勾配降下
final_pos, path, steps = core.roll_marble(unknown_vec)
```
