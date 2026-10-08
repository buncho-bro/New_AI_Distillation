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

## 🚀 What is this Method? (The Constellation Method)
従来のAI蒸留（Distillation）は、巨大な教師モデルに大量のプロンプトを入力し、その出力（Logits）を生徒モデルに学習させるため、**膨大な計算資源（VRAMとGPU時間）**を必要としました。

本プロジェクトが提唱する **"Topology Distillation" (概念地形蒸留)**、別名 **"The Constellation Method" (星座方式)** は、その常識を覆す手法です。

1. **VRAMゼロでの知識抽出**: 教師モデル（何百GBもある巨大モデル）の推論エンジンを起動せず、最初の辞書部分（`embed_tokens.weight`）だけを CPU で直接読み込みます。
2. **次元の壁の突破**: 生徒モデル（例: Qwen-3B, 1536次元）と教師モデル（例: GLM-5.3, 6144次元）では、脳の次元（ベクトルの長さ）が異なるため、直接的な比較は不可能です。
3. **星座の形（N x N Matrix）**: そこで、概念の絶対座標ではなく、「『変数』と『関数』の距離」といった概念同士の**「相対的な距離行列（N x N）」**を計算します。次元が異なっても「星座の形」は同じように計算できるため、この距離行列をTargetとして生徒モデルに LoRA で焼き付け（MSE Loss）ます。

これにより、一般家庭のPC（VRAM 12GB程度）でも、最先端の巨大モデルの「概念構造」を小さなモデルへ短時間で移植することが可能になりました。

---

## 💡 Why this Method? (なぜこの方法になったのか)
開発の初期段階では、「教師モデルから抽出したベクトルをJSONで書き出し、推論時にそれを直接足し込む（Logit Bias）」という簡易的なアプローチを試みました。
しかし、この方法には致命的な欠陥がありました。**「次元の壁」**です。

1536次元の生徒モデルの空間に、6144次元の教師モデルのベクトルを強制的に干渉させようとしたため、テンソルのサイズ不一致エラー（`Tensor size mismatch`）を引き起こし、システムがクラッシュしました。

この壁を乗り越えるため、「ベクトルそのもの」を注入するのではなく、**「単語と単語の距離感（相対関係＝星座の形）」を算出し、それを LoRA (Low-Rank Adaptation) を使って生徒モデルの重みとして PyTorch 上で最適化（学習）させる** という、より本質的で強固な手法へと進化しました。これが現在の方法です。

---

## 📂 File Structure (主要ファイルの説明)

*   **`distillation_experiment/auto_shallow_distiller.py`**
    *   **役割**: 本プロジェクトの中核となる「星座蒸留」を自動実行するスクリプト。
    *   **内容**: 教師モデル（GLM等）からVRAM消費ゼロで Embedding を抽出し、相対距離行列（Target Topology）を生成。その後、生徒モデル（Qwen等）をロードし、`optim.Adam` を使って星座の形を正規の LoRA アダプタ（`adapter_model.safetensors`）として焼き付けます。
*   **`distillation_experiment/run_distilled_benchmark.py`**
    *   **役割**: 蒸留後のモデルのコーディング能力を測定するベンチマークスクリプト。
    *   **内容**: HumanEval（164問のPythonテスト）をダウンロードし、生成したコードをローカルの簡易サンドボックスで実行（`exec`）して Pass@1 の正答率を算出します。
*   **`project_ripple/` (波紋モデルUI)**
    *   **役割**: 蒸留されたモデルや、概念物理エンジンを体験するためのチャットUIアプリケーション。
    *   **内容**: `server.py` (FastAPIバックエンド) と HTML5 Canvas を用いた Diorama UI を搭載。`physics_core.py` はロードされた LoRA を認識し、引力や粘性といった物理法則を推論プロセスに適用します。
*   **`concept_forge_webui.py`**
    *   **役割**: 概念抽出と蒸留を視覚的に行うための Gradio ベースの実験用統合UI。

---

## 📖 Terminology (用語集)

*   **Concept Topography (概念地形)**
    *   AIの「脳内（潜在空間）」に広がる、単語や概念の配置図のこと。優れたAIほど、この地形が美しく整理されています。
*   **Shallow Extraction (浅層抽出 / No VRAM Teacher)**
    *   巨大なAIモデルの重いTransformer層を起動せず、表面にある「辞書ファイル（Embeddings層）」だけをVRAM消費ゼロ（CPUのみ）で直接抜き取る技術。
*   **The Constellation Method (星座方式)**
    *   異なる次元のAI同士で知識を同期させるため、絶対座標ではなく「概念同士の相対距離（星と星の距離）」を使って比較・焼き付けを行う手法。
*   **Gravity / Viscosity (引力 / 粘性)**
    *   Project Ripple (Chatアプリ) において、AIの思考（トークン生成）を特定の概念へ引っ張る力（Gravity）と、その力を受けて思考がどれだけブレーキをかけられるか（Viscosity）を定義する物理パラメーター。
