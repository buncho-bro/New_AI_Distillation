import os
import pickle
import time
import torch
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FAISS_FULL_PATH = os.path.join(OUTPUT_DIR, "wikipedia_full.faiss")
TEXTS_PATH = os.path.join(OUTPUT_DIR, "wikipedia_texts.pkl")

def build_reverse_mapping_db(num_samples=5000):
    print("="*60)
    print("Building Reverse-Mapping Database (Approach B Test)")
    print("="*60)
    
    # 1. データの取得
    print(f"Fetching {num_samples} articles from Wikipedia...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
    texts = []
    for i, item in enumerate(dataset):
        if i >= num_samples: break
        text = item['text'].split('\n')[0]
        if len(text) > 30:
            texts.append(text)
            
    print(f"Loaded {len(texts)} valid texts.")
    
    # 2. エンコード
    print("Encoding texts...")
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = embeddings.astype(np.float32)
    
    # 3. FAISSへの全件登録
    print("Building FAISS Index for actual texts...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)  # コサイン類似度空間での距離計算用
    index.add(embeddings)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_FULL_PATH)
    
    # 4. テキストデータの保存（FAISSのIDと紐付け）
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
        
    print(f"[SUCCESS] Saved {len(texts)} texts and vectors for Reverse-Mapping!")

def simulate_concept_decoding(query="最新の宇宙開発について"):
    print(f"\n--- Simulating Approach B Decoder ---")
    print(f"User Query: '{query}'")
    
    # 本来はここでPentagonの重力・熱トレースの計算が入るが、
    # 今回はシンプルにクエリベクトルを「到達した概念座標」とする
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    query_vec = encoder.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_vec)
    
    # データベースのロード
    index = faiss.read_index(FAISS_FULL_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
        
    start_time = time.time()
    
    # 1. 逆写像 (Reverse-Mapping): 到達座標から周囲の概念の破片を Top-10 引き上げる
    k_fragments = 10
    distances, indices = index.search(query_vec, k_fragments)
    
    print(f"\n[Extraction Time]: {time.time() - start_time:.4f} seconds")
    print(f"--- Extracted Concept Fragments (Raw Knowledge) ---")
    
    fragments = []
    for i, idx in enumerate(indices[0]):
        frag = texts[idx][:100] + "..."  # 長すぎるので省略表示
        fragments.append(frag)
        print(f"Fragment {i+1} (Dist: {distances[0][i]:.3f}): {frag}")
        
    # 2. 翻訳機構（LLM）への命令構築
    print("\n--- Constructing Prompt for Translator (LLM) ---")
    prompt = (
        "あなたは概念から言語を紡ぐ『翻訳機』です。自身の知識（事前学習）は使わず、"
        "以下の【概念の破片】だけを繋ぎ合わせ、ユーザーの意図に沿った自然な日本語に再構築してください。\n\n"
        "【ユーザーの意図】\n"
        f"{query}\n\n"
        "【概念の破片】\n"
    )
    for frag in fragments:
        prompt += f"- {frag}\n"
        
    print(prompt)
    print("==================================================")
    print("（※ここで上記のプロンプトをLLM APIに渡し、最終的な自然言語を生成させます）")

if __name__ == "__main__":
    build_reverse_mapping_db(5000)
    simulate_concept_decoding("人工知能と最新のテクノロジーについて")
