import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans, MeanShift
import warnings
warnings.filterwarnings("ignore")

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MassiveConceptEngine:
    def __init__(self, dim=384):
        self.dim = dim
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.anchors = None
        
        # 巨大空間では谷が密集するため、引力の範囲(sigma)を狭め(0.2)にして干渉を防ぐ
        p = HDParams(
            dim=self.dim, use_ridge=False,
            sigma=0.2, sigma_rep=1.0, alpha_rep=0.5,
            marble_lr=0.5, marble_steps=150,  # 巨大地形ではステップ数を多めにする
            marble_max_v=1.0, marble_momentum=0.2
        )
        self.core = HighDimConceptCore(p)

    def discover_massive_concepts(self, raw_texts):
        print(f"Injecting {len(raw_texts)} real-world texts into the vector space...")
        start_t = time.time()
        embeddings = self.encoder.encode(raw_texts, show_progress_bar=True, batch_size=128)
        print(f"Encoding took {time.time() - start_t:.2f} seconds.")
        
        print("\nRunning clustering to discover dense concept valleys...")
        start_t = time.time()
        # 巨大データに対してMeanShiftは遅すぎるため、MiniBatchKMeansで多数の細かい谷（マイクロクラスタ）を形成させる
        # AI自身に細かい地形を掘らせるイメージ
        n_valleys = 100 # 100個の谷（概念のクラスタ）を自動生成
        clustering = MiniBatchKMeans(n_clusters=n_valleys, random_state=42, batch_size=512)
        clustering.fit(embeddings)
        print(f"Clustering took {time.time() - start_t:.2f} seconds.")
        
        centroids = clustering.cluster_centers_
        self.anchors = torch.tensor(centroids).float()
        self.anchors = self.anchors / self.anchors.norm(dim=1, keepdim=True)
        
        self.core._centers = self.anchors.clone()
        # クラスタに含まれるデータ数(サイズ)を谷の深さ(重み)として設定
        _, counts = np.unique(clustering.labels_, return_counts=True)
        weights = torch.tensor(counts).float()
        # 重みを正規化（最大を2.0くらいにする）
        self.core._weights = (weights / weights.max()) * 2.0
        
        print(f" => Formed a massive landscape with {n_valleys} valleys of varying depths.")
        return embeddings, clustering.labels_

    def run_inference(self, user_text):
        emb = self.encoder.encode([user_text])
        x = torch.tensor(emb).float().squeeze(0)
        x = x / x.norm()
        x = x.requires_grad_(True)
        
        v = torch.zeros_like(x)
        path = [x.detach().clone()]
        
        start_t = time.time()
        for i in range(self.core.p.marble_steps):
            H = self.core.potential(x.unsqueeze(0)).squeeze()
            H.backward()
            grad = x.grad.detach()
            x.grad.zero_()
            
            v = self.core.p.marble_momentum * v - self.core.p.marble_lr * grad
            v_norm = v.norm()
            if v_norm > self.core.p.marble_max_v:
                v = v * (self.core.p.marble_max_v / v_norm)
            
            x_new = x.detach() + v
            x_new = x_new / x_new.norm()
            x = x_new.requires_grad_(True)
            path.append(x.detach().clone())
            
            # 早めに収束(速度がほぼ0)したらブレイク(計算量節約)
            if v_norm < 1e-4:
                print(f"    [Physics] Marble stabilized at step {i}")
                break
                
        inf_time = time.time() - start_t
        
        final_pos = x.detach()
        dists = torch.norm(self.core._centers - final_pos.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(dists).item()
        min_dist = dists[nearest_idx].item()
        
        return nearest_idx, min_dist, path, inf_time

import random

def generate_synthetic_news(n=3000):
    topics = {
        'tech': (['Apple', 'Microsoft', 'Google', 'AI', 'Software'], ['released', 'announced', 'updated', 'developed'], ['a new smartphone', 'an AI model', 'a cloud service', 'an operating system']),
        'space': (['NASA', 'SpaceX', 'Astronauts', 'The rover', 'A telescope'], ['launched', 'discovered', 'landed on', 'photographed'], ['a new planet', 'Mars', 'the moon', 'a black hole', 'the space station']),
        'politics': (['The president', 'The senator', 'The government', 'The committee'], ['signed', 'vetoed', 'discussed', 'proposed'], ['a new law', 'the budget', 'a trade agreement', 'the healthcare bill']),
        'sports': (['The team', 'The striker', 'The athlete', 'The champion'], ['won', 'lost', 'scored', 'defended'], ['the gold medal', 'the world cup', 'the final match', 'the championship title'])
    }
    
    texts = []
    for _ in range(n):
        topic = random.choice(list(topics.keys()))
        s, v, o = topics[topic]
        text = f"{random.choice(s)} {random.choice(v)} {random.choice(o)}."
        # Add some random noise words to make it slightly more complex
        noise = [" Yesterday.", " In a surprising turn of events.", " Experts say this is huge.", " The public is waiting."]
        text += random.choice(noise)
        texts.append(text)
    return texts

def run_massive_scenario():
    print("Generating ~3000 synthetic news articles...")
    texts = generate_synthetic_news(3000)
    print(f"Generated {len(texts)} valid texts.")
    
    engine = MassiveConceptEngine(dim=384)
    
    print("\n==========================================")
    print(" Massive Landscape Construction")
    print("==========================================")
    raw_embs, labels = engine.discover_massive_concepts(texts)
    
    test_queries = [
        "What is the best software for 3D rendering and animation?",
        "NASA is launching a new telescope into orbit next year.",
        "The president discussed the peace treaty in the Middle East.",
        "My car engine is making a weird noise when I accelerate."
    ]
    
    print("\n==========================================")
    print(" Inference on Massive Landscape")
    print("==========================================")
    
    query_paths = []
    total_inf_time = 0
    for q in test_queries:
        print(f"\n[Query]: '{q}'")
        idx, min_dist, path, inf_time = engine.run_inference(q)
        total_inf_time += inf_time
        print(f"  => Stopped at Valley #{idx} (Depth: {engine.core._weights[idx].item():.2f})")
        print(f"  => Distance to bottom: {min_dist:.4f}")
        print(f"  => Inference time: {inf_time:.4f} sec, Steps taken: {len(path)}")
        query_paths.append((q, path))
        
    print(f"\nAverage Inference Time per query: {total_inf_time / len(test_queries):.4f} sec")
    
    # 巨大ランドスケープの一部をPCA可視化 (データが多すぎるので間引く)
    print("\nVisualizing Massive Landscape...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # 重心とクエリ軌跡だけでPCAの軸を決める（全データを入れるとノイズが大きすぎるため）
    pca_fit_data = torch.cat([engine.anchors] + [torch.stack(p) for _, p in query_paths]).numpy()
    pca.fit(pca_fit_data)
    
    anchors_2d = pca.transform(engine.anchors.numpy())
    
    plt.figure(figsize=(12, 10))
    
    # 谷のプロット（深さに応じてサイズを変える）
    weights_np = engine.core._weights.numpy()
    plt.scatter(anchors_2d[:, 0], anchors_2d[:, 1], s=weights_np*300, color='gray', alpha=0.5, edgecolor='black', label='Discovered Valleys')
    
    # 最も深いトップ5の谷を赤く塗る
    top_indices = np.argsort(weights_np)[-5:]
    for i in top_indices:
        plt.scatter(anchors_2d[i, 0], anchors_2d[i, 1], s=weights_np[i]*300, color='red', edgecolor='black', zorder=4)
        plt.text(anchors_2d[i, 0], anchors_2d[i, 1]-0.05, f"Deep Valley #{i}", fontsize=10, weight='bold', ha='center')
        
    # クエリの軌跡
    colors = ['blue', 'green', 'purple', 'orange']
    for i, (q, path) in enumerate(query_paths):
        p_2d = pca.transform(torch.stack(path).numpy())
        plt.plot(p_2d[:, 0], p_2d[:, 1], color=colors[i], linestyle='-', linewidth=2, zorder=5)
        plt.scatter(p_2d[0, 0], p_2d[0, 1], color=colors[i], s=150, marker='P', edgecolor='black', label=f"Query {i+1}", zorder=6)
        # 終点
        plt.scatter(p_2d[-1, 0], p_2d[-1, 1], color=colors[i], s=200, marker='X', edgecolor='black', zorder=6)
        
    plt.title("Massive Concept Landscape (100 Valleys from ~3000 Texts)", fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "massive_landscape.png")
    plt.savefig(out_path, dpi=150)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    run_massive_scenario()
