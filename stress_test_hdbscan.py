import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import gc

def load_base_data():
    import random
    topics = {
        'Tech': (['Apple', 'Microsoft'], ['released', 'updated'], ['a smartphone', 'an OS']),
        'Space': (['NASA', 'SpaceX'], ['launched', 'discovered'], ['a planet', 'a rocket']),
        'Politics': (['The president', 'The senator'], ['signed', 'vetoed'], ['a law', 'the budget']),
        'Sports': (['The team', 'The athlete'], ['won', 'lost'], ['the medal', 'the match']),
        'Food': (['The chef', 'My mother'], ['cooked', 'baked'], ['a pizza', 'a cake']),
        'Music': (['The band', 'The singer'], ['performed', 'released'], ['a song', 'an album'])
    }
    texts = []
    for topic_name, parts in topics.items():
        s, v, o = parts
        for _ in range(100):
            text = f"{random.choice(s)} {random.choice(v)} {random.choice(o)}."
            texts.append(text)
    return texts

def run_stress_test():
    print("--- HDBSCAN Stress Test ---")
    texts = load_base_data()
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("Encoding base vectors...")
    base_embeddings = encoder.encode(texts, show_progress_bar=False)
    
    # 段階的に増やすデータ数
    test_sizes = [10000, 50000, 100000, 200000, 500000]
    
    for size in test_sizes:
        print(f"\n[Test] N = {size:,} points")
        
        # ベースベクトルを複製し、微小ノイズ(標準偏差0.05)を加えて大量のデータを生成
        repeats = (size // len(base_embeddings)) + 1
        expanded = np.tile(base_embeddings, (repeats, 1))[:size]
        noise = np.random.normal(0, 0.05, expanded.shape).astype(np.float32)
        test_data = expanded + noise
        
        print(f"Data generated. Running HDBSCAN...")
        start_time = time.time()
        
        try:
            # メモリを大量に消費するため、ガーベジコレクションを強制
            gc.collect()
            
            clusterer = HDBSCAN(min_cluster_size=20, metric='euclidean')
            labels = clusterer.fit_predict(test_data)
            
            elapsed = time.time() - start_time
            unique_labels = set(labels)
            n_noise = (labels == -1).sum()
            if -1 in unique_labels:
                unique_labels.remove(-1)
                
            print(f"-> Time: {elapsed:.2f} seconds")
            print(f"-> Valleys Found: {len(unique_labels)}")
            print(f"-> Noise Points: {n_noise:,}")
            
            if len(unique_labels) <= 1:
                print(">>> BREAKDOWN DETECTED: All distinct concepts collapsed into a single massive valley (or zero)! Dimensionality curse occurred.")
                break
                
        except MemoryError:
            print(f">>> BREAKDOWN DETECTED: OUT OF MEMORY (OOM) at N={size:,}.")
            break
        except Exception as e:
            print(f">>> BREAKDOWN DETECTED: Algorithm crashed: {e}")
            break
            
if __name__ == "__main__":
    run_stress_test()
