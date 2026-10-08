import os
import time
import torch
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN, MiniBatchKMeans

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "pentagon_landscape_massive.faiss")
META_PATH = os.path.join(OUTPUT_DIR, "pentagon_landscape_massive_meta.pt")

def fetch_wikipedia_data(num_samples=50000):
    print(f"Fetching {num_samples} articles from Japanese Wikipedia...")
    # ストリーミングモードで重いデータのダウンロードを回避し、必要な分だけ取得
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
    
    texts = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        
        # 記事の冒頭（要約部分）を抽出
        text = item['text'].split('\n')[0]
        # 短すぎるものはスキップ
        if len(text) > 30:
            texts.append(text)
            
    print(f"Successfully loaded {len(texts)} articles.")
    return texts

def train_massive_landscape():
    print("="*60)
    print("Phase 1: Building Massive Concept Landscape (Wikipedia)")
    print("="*60)
    
    # 1. データ取得
    start_time = time.time()
    texts = fetch_wikipedia_data(50000)
    print(f"Data fetch time: {time.time() - start_time:.2f}s")
    
    # 2. エンコード
    print("\nEncoding documents into vector space (This may take a few minutes)...")
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = embeddings.astype(np.float32)
    
    # 3. HDBSCANによる「最適な谷の数」の自動算出（ハイブリッド戦略）
    # 5万件すべてをやると破綻するため、1万件をサンプリング
    print("\nRunning HDBSCAN on 10,000 samples to autonomously determine optimal K...")
    sample_size = min(10000, len(embeddings))
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[indices]
    
    clusterer = HDBSCAN(min_cluster_size=20, metric='euclidean')
    sample_labels = clusterer.fit_predict(sample_embeddings)
    unique_labels = set(sample_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
        
    sample_k = len(unique_labels)
    print(f"-> HDBSCAN found {sample_k} distinct valleys in {sample_size} samples.")
    
    # 標本比率から全体の最適なKを推測（単純比例ではなく、対数的な増加を仮定）
    # K_total = K_sample * log(N_total) / log(N_sample)
    optimal_k = int(sample_k * (np.log(len(embeddings)) / np.log(sample_size)))
    # 最低でも100、最大でも2000にクリップ
    optimal_k = max(100, min(optimal_k, 2000))
    print(f"-> Calculated optimal valleys (K) for all {len(embeddings):,} data points: {optimal_k}")
    
    # 4. 全データの高速クラスタリング (MiniBatchKMeans)
    print(f"\nRunning MiniBatchKMeans (K={optimal_k}) on ALL {len(embeddings):,} points...")
    kmeans = MiniBatchKMeans(n_clusters=optimal_k, batch_size=1024, max_iter=100, random_state=42)
    kmeans.fit(embeddings)
    
    # 各谷の重み（深さ）を計算
    labels = kmeans.labels_
    anchors = kmeans.cluster_centers_
    weights = np.zeros(optimal_k, dtype=np.float32)
    for i in range(optimal_k):
        weights[i] = np.sum(labels == i)
        
    # 重みを正規化
    weights = (weights / weights.max()) * 2.0
    
    # 5. FAISS インデックスの構築
    print("\nBuilding FAISS Index for ultra-fast concept gravity search...")
    dimension = anchors.shape[1]
    # L2距離用のインデックス（内積の場合は IndexFlatIP）
    index = faiss.IndexFlatL2(dimension)
    
    # FAISSに追加する前にベクトルを正規化（Cosine Similarity空間にする場合）
    faiss.normalize_L2(anchors)
    index.add(anchors)
    
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    print(f"Saving metadata to {META_PATH}")
    torch.save({
        'weights': torch.tensor(weights),
        'k': optimal_k,
        'dimension': dimension,
        'total_data_points': len(embeddings)
    }, META_PATH)
    
    print(f"\n[SUCCESS] Massive Wikipedia Landscape successfully built and saved!")

if __name__ == "__main__":
    train_massive_landscape()
