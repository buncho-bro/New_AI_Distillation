# New AI Distillation (Concept Forge / Project Ripple)

This project explores a novel approach to AI model distillation ("Topology Distillation" / "The Constellation Method"), enabling the transfer of structural concept representations (Concept Topography) from massive LLMs to highly efficient, lightweight models without the need for massive VRAM overhead.

## 📌 Project Philosophy & Disclaimers

1. **Created with Google Antigravity**
   This project, including its core algorithms, scripts, and UI architecture, was primarily developed in collaboration with **Gemini Pro** using the **Google Antigravity** AI agent framework.

2. **Aiming for Fair AI Competition**
   The primary goal of this project is to foster **fair competition and democratization in AI development**. By allowing developers to extract and distill "concept topographies" locally and efficiently, we hope to lower the barrier to entry for AI research. This project was **never intended to disrupt, exploit, or destroy existing AI systems**. It is an academic and engineering pursuit for optimization and equality.

3. **Compliance with Original AI Licenses**
   When utilizing the scripts in this repository to extract knowledge or distill models, **you must strictly adhere to the terms of service, licenses, and usage rules provided by the original creators of the respective AI models** (e.g., Hugging Face, OpenAI, Z.ai, Alibaba, etc.). We do not condone the use of these tools for copyright infringement or violation of specific model licenses.

---

## 💡 Evolution of the Methodology (なぜこの手法に行き着いたのか)

本プロジェクトの開発過程は、いくつもの失敗と仮説の検証を経て、現在の洗練された手法へと到達しました。

### 1. なぜ通常の蒸留（Knowledge Distillation）を使わないのか？
従来の蒸留は、巨大な教師モデル（Teacher）に膨大なプロンプトを入力し、その出力（Logitsや生成テキスト）を生徒モデル（Student）に学習させます。しかしこの方法では、**「教師モデルを動かすための莫大なVRAM（何百GB）」**と**「推論し続けるための膨大なGPU時間」**が必要であり、個人や小規模チームでの実行は不可能でした。
そこで、「モデル全体を動かすのではなく、**AIの脳の表面にある辞書（概念の初期配置）だけを抜き取れないか？**」という発想が生まれました。

### 2. 簡易JSON抽出の失敗と「次元の壁」
初期のアプローチでは、教師モデル（GLM-5.3等）の Embeddings 層からベクトルを抜き出し、単なる辞書データ（JSON）として保存。それを推論時に Logit Bias として足し込む方法を試みました。
しかし、このアプローチは**「次元の壁」**に衝突し、クラッシュしました。教師モデル（6144次元）と生徒モデル（1536次元や3072次元）では脳の構造が全く異なるため、ベクトルを直接比較したり注入したりすることは数学的に不可能だったのです。

### 3. The Constellation Method（星座方式）の誕生
次元の壁を越えるため、私たちは「絶対座標（ベクトルそのもの）」を諦め、**「相対座標（概念同士の距離）」**に着目しました。
「関数」と「変数」の距離、「ネットワーク」と「プロトコル」の距離など、概念と概念の相対的な位置関係（星座の形）であれば、次元が異なっても完全に一致させることができます。
この「星座の形」を抽出して生徒モデルに LoRA で焼き付けるという、極めてエレガントな手法 **"The Constellation Method" (星座方式)** がここに完成しました。

---

## 🚀 Technical Deep Dive (各技術の詳細な仕組み)

### A. CPU-Only Shallow Extraction (浅層抽出)
巨大モデルから知識を抽出する際、モデルを PyTorch (`AutoModel.from_pretrained`) で初期化すると即座にVRAMが枯渇（OOM）します。
これを回避するため、HuggingFaceの `safetensors` ライブラリの **`safe_open`** を使用します。これにより、モデル全体の計算グラフを構築することなく、ディスク上から `embed_tokens.weight`（単語の辞書ベクトル）だけを **CPUのRAM上へ直接・部分的に読み込む** ことができます。教師モデルに対する VRAM 消費は完全にゼロ（0MB）です。

### B. Topological Distillation (概念地形蒸留)
抽出した教師ベクトルを、生徒に学習（焼き付け）させる詳細なプロセスです。
1. **Target Topology の生成**: 教師モデルから抽出した $N$ 個の概念ベクトル（6144次元など）に対し、ベクトル同士のコサイン類似度や内積を計算します。これにより、**$N \times N$ の距離行列（相対座標マトリックス）** が生成されます。これが「目標となる星座の形」です。
2. **Student Topology の生成**: 生徒モデル（例: Qwen-3B + LoRA）に同じ $N$ 個の単語を入力し、最終隠れ層（Hidden States）からベクトル（1536次元など）を取得。同様に $N \times N$ の距離行列を計算します。
3. **Lossの最適化**: 2つの $N \times N$ 行列に対し、**MSE (平均二乗誤差) Loss** を計算し、`optim.Adam` で生徒の LoRA ウェイトを更新します。次元の違いを完全に無視し、「思考の形（概念の結びつきの強さ）」だけをコピーする画期的なファインチューニングです。

### C. Project Ripple (概念物理エンジン)
蒸留された知識をチャットUIで視覚的・物理的に活用するための独自の推論システムです。
HuggingFace の `LogitsProcessor` を拡張した `RippleTopographyProcessor` を実装しています。
*   **Gravity (引力)**: 次のトークンを予測する際、指定されたターゲット概念（ベクトル）との内積（アフィニティ）を計算し、ロジット（確率）に加算して出力を引っ張ります。
*   **Viscosity & Decay (粘性と減衰)**: 単に引っ張るだけでは文章が壊れるため、生成が進む（ステップが進む）につれて引力を指数関数的に弱める（Decay）処理と、波紋のように滑らかに適用する粘性（Viscosity）パラメータを組み込み、自然な会話のまま特定ドメインの知識を引き出すことを可能にしています。

---

## 📂 File Structure (主要ファイル)

*   **`distillation_experiment/auto_shallow_distiller.py`**
    *   教師モデルからVRAMゼロで Embedding を抽出し、N×Nの距離行列（Target Topology）を生成。生徒モデルをロードし、`optim.Adam` を使って星座の形を正規の LoRA アダプタとして焼き付けるメインスクリプト。
*   **`distillation_experiment/run_evalplus_benchmark.py`**
    *   蒸留後のコーディング能力を測定するベンチマーク。HumanEval+ (EvalPlus) の過酷な境界値テストをローカルのサンドボックスで実行（無限ループ回避のマルチプロセス・タイムアウト付き）し、スコアを算出します。
*   **`project_ripple/` (波紋モデルUI)**
    *   FastAPIバックエンドと HTML5 Canvas を用いたチャットアプリケーション。`physics_core.py` によって LoRA を読み込み、Gravity / Viscosity の物理法則を推論に適用します。

---

## 📖 Terminology (用語集)

*   **Concept Topography (概念地形)**
    *   AIの「脳内（潜在空間）」に広がる、単語や概念の配置図のこと。優れたAIほど、この地形が美しく整理されています。
*   **Shallow Extraction (浅層抽出 / No VRAM Teacher)**
    *   巨大なAIモデルの重いTransformer層を起動せず、表面にある「辞書ファイル（Embeddings層）」だけをVRAM消費ゼロ（CPUのみ）で直接抜き取る技術。
*   **The Constellation Method (星座方式)**
    *   異なる次元のAI同士で知識を同期させるため、絶対座標ではなく「概念同士の相対距離（星と星の距離）」を使って比較・焼き付けを行う手法。
