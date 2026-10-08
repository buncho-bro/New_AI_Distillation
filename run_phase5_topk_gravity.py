import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings("ignore")

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 解決策: Top-K Gravity (アテンション重力) を搭載したカスタムエンジン
# -----------------------------------------------------------------
class TopKConceptCore(HighDimConceptCore):
    def __init__(self, p, k=3):
        super().__init__(p)
        self.k = k # 感知する谷の数

    def potential(self, x):
        """
        現在地 x から近い上位 K 個の谷の引力だけを計算し、干渉を防ぐ。
        """
        # x: (1, dim), self._centers: (N, dim)
        dists = torch.norm(self._centers - x, dim=1) # (N,)
        
        # 谷が K 個未満の場合はすべて使用
        k_actual = min(self.k, len(self._centers))
        
        # 距離が近い上位 K 個のインデックスを取得
        topk_vals, topk_indices = torch.topk(dists, k_actual, largest=False)
        
        H = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for idx in topk_indices:
            dist_sq = dists[idx]**2
            # 解決策2: ガウス関数(exp)による勾配消失を防ぐため、
            # 物理的なバネの力（フックの法則 / 調和振動子）のエネルギーに変更！
            # 遠く離れていても、最も近いTop-Kの谷からは確実に引っ張られる。
            H += 0.5 * self._weights[idx] * dist_sq
            
        return H

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
        noise = [" Yesterday.", " In a surprising turn of events.", " Experts say this is huge.", " The public is waiting."]
        text += random.choice(noise)
        texts.append(text)
    return texts

class SolvedMassiveEngine:
    def __init__(self, dim=384):
        self.dim = dim
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.anchors = None
        
        # 遠くの干渉をカットしたので、引力が届く範囲(sigma)を広く大胆に設定できる！
        # これにより、確実に谷の底まで引きずり込めるようになる
        p = HDParams(
            dim=self.dim, use_ridge=False,
            sigma=0.6,  # 以前は干渉を恐れて0.2だったが、0.6まで広げる
            sigma_rep=1.0, alpha_rep=0.5,
            marble_lr=0.8, marble_steps=200, 
            marble_max_v=1.0, marble_momentum=0.4
        )
        # Top-1 Gravity (Softmax/Winner-takes-all) を使用。
        # 複数に引かれると中間で釣り合ってしまうため、最も近い1つの概念に全振りする
        self.core = TopKConceptCore(p, k=1)

    def discover_massive_concepts(self, raw_texts):
        print(f"Injecting {len(raw_texts)} real-world texts...")
        embeddings = self.encoder.encode(raw_texts, show_progress_bar=False, batch_size=256)
        
        print("Clustering 100 valleys...")
        n_valleys = 100
        clustering = MiniBatchKMeans(n_clusters=n_valleys, random_state=42, batch_size=512)
        clustering.fit(embeddings)
        
        centroids = clustering.cluster_centers_
        self.anchors = torch.tensor(centroids).float()
        self.anchors = self.anchors / self.anchors.norm(dim=1, keepdim=True)
        
        self.core._centers = self.anchors.clone()
        _, counts = np.unique(clustering.labels_, return_counts=True)
        weights = torch.tensor(counts).float()
        self.core._weights = (weights / weights.max()) * 2.0
        
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
            
            if v_norm < 1e-4 and i > 5:
                print(f"    [Physics] Marble reached the bottom at step {i}")
                break
                
        inf_time = time.time() - start_t
        
        final_pos = x.detach()
        dists = torch.norm(self.core._centers - final_pos.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(dists).item()
        min_dist = dists[nearest_idx].item()
        
        return nearest_idx, min_dist, path, inf_time

def run_solved_scenario():
    print("Generating 3000 synthetic news articles...")
    texts = generate_synthetic_news(3000)
    
    engine = SolvedMassiveEngine(dim=384)
    engine.discover_massive_concepts(texts)
    
    test_queries = [
        "What is the best software for 3D rendering and animation?",
        "NASA is launching a new telescope into orbit next year.",
        "The president discussed the peace treaty in the Middle East.",
        "My car engine is making a weird noise when I accelerate."
    ]
    
    print("\n==========================================")
    print(" Inference with Top-K Gravity (Attention)")
    print("==========================================")
    
    query_paths = []
    for q in test_queries:
        print(f"\n[Query]: '{q}'")
        idx, min_dist, path, inf_time = engine.run_inference(q)
        print(f"  => Stopped at Valley #{idx} (Depth: {engine.core._weights[idx].item():.2f})")
        print(f"  => Distance to bottom: {min_dist:.4f}")
        print(f"  => Inference time: {inf_time:.4f} sec, Steps taken: {len(path)}")
        if len(path) > 10 and min_dist < 0.6:
            print("  [SUCCESS] The marble rolled smoothly into the valley!")
        else:
            print("  [FAILED] Marble is still stuck.")
        query_paths.append((q, path))
        
    # PCA可視化
    print("\nVisualizing Solved Landscape...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_fit_data = torch.cat([engine.anchors] + [torch.stack(p) for _, p in query_paths]).numpy()
    pca.fit(pca_fit_data)
    
    anchors_2d = pca.transform(engine.anchors.numpy())
    plt.figure(figsize=(12, 10))
    weights_np = engine.core._weights.numpy()
    plt.scatter(anchors_2d[:, 0], anchors_2d[:, 1], s=weights_np*300, color='gray', alpha=0.5, edgecolor='black', label='Discovered Valleys')
    
    colors = ['blue', 'green', 'purple', 'orange']
    for i, (q, path) in enumerate(query_paths):
        p_2d = pca.transform(torch.stack(path).numpy())
        plt.plot(p_2d[:, 0], p_2d[:, 1], color=colors[i], linestyle='-', linewidth=3, zorder=5)
        plt.scatter(p_2d[0, 0], p_2d[0, 1], color=colors[i], s=150, marker='P', edgecolor='black', label=f"Query {i+1}", zorder=6)
        plt.scatter(p_2d[-1, 0], p_2d[-1, 1], color=colors[i], s=200, marker='X', edgecolor='black', zorder=6)
        
    plt.title("Massive Concept Landscape (Solved with Top-K Gravity)", fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "solved_landscape.png")
    plt.savefig(out_path, dpi=150)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    run_solved_scenario()
