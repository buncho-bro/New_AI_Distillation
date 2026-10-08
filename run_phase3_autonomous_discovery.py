import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MeanShift
import warnings
warnings.filterwarnings("ignore")

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AutonomousConceptEngine:
    def __init__(self, dim=384):
        self.dim = dim
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # アンカー（谷）は最初は空っぽ
        self.anchors = None
        self.domain_texts = []
        
        p = HDParams(
            dim=self.dim, use_ridge=False,
            sigma=0.3, sigma_rep=1.0, alpha_rep=0.5,
            marble_lr=0.5, marble_steps=100,
            marble_max_v=1.0, marble_momentum=0.2
        )
        self.core = HighDimConceptCore(p)

    def discover_concepts(self, raw_texts):
        print(f"Injecting {len(raw_texts)} unlabeled data points into the space...")
        # 1. すべてのテキストをベクトル化（砂粒をばら撒く）
        embeddings = self.encoder.encode(raw_texts)
        
        print("Running MeanShift algorithm for autonomous density discovery...")
        # 2. MeanShiftで「密度のピーク（重心）」を自動発見（クラスター数も自動決定）
        clustering = MeanShift(bandwidth=0.5).fit(embeddings)
        centroids = clustering.cluster_centers_
        n_clusters = len(centroids)
        print(f" => AI automatically discovered {n_clusters} natural concept valleys!")
        
        # 3. 発見された重心を概念核のアンカー（谷）に設定
        self.anchors = torch.tensor(centroids).float()
        self.anchors = self.anchors / self.anchors.norm(dim=1, keepdim=True)
        
        self.core._centers = self.anchors.clone()
        self.core._weights = torch.ones(len(self.anchors)) * 1.0
        
        # 発見された谷に、AI目線での仮のIDを割り振る
        self.domain_texts = [f"Autonomously Discovered Valley #{i}" for i in range(n_clusters)]
        
        return embeddings, clustering.labels_

    def run_inference(self, user_text):
        emb = self.encoder.encode([user_text])
        x = torch.tensor(emb).float().squeeze(0)
        x = x / x.norm()
        x = x.requires_grad_(True)
        
        v = torch.zeros_like(x)
        path = [x.detach().clone()]
        
        for _ in range(self.core.p.marble_steps):
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
            
        final_pos = x.detach()
        dists = torch.norm(self.core._centers - final_pos.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(dists).item()
        return nearest_idx, path

def run_autonomous_scenario():
    # 完全に未分類の雑多なデータ（ラベルなし）
    unlabeled_data = [
        # 宇宙系っぽいデータ
        "The rocket launched into orbit.", "NASA discovered a new exoplanet.", 
        "Mars rover sends back images.", "Spacecraft docked at ISS.", 
        "Astronauts walk on the moon.", "A massive black hole was found.",
        # 料理系っぽいデータ
        "The chef prepared a delicious pasta.", "Baking a chocolate cake.", 
        "Fresh ingredients make the best salad.", "Cooking dinner for the family.", 
        "A recipe for spicy curry.", "Grilling steak on the barbecue.",
        # スポーツ系っぽいデータ
        "The striker scored a beautiful goal.", "The team won the championship match.", 
        "Tennis final went to five sets.", "Basketball player dunked the ball.", 
        "Olympic gold medal won by swimmer.", "A hole-in-one at the golf tournament."
    ]
    
    engine = AutonomousConceptEngine(dim=384)
    
    # AI自身に地形を形成させる
    print("\n==========================================")
    print(" Phase 3.1: Autonomous Concept Discovery")
    print("==========================================")
    raw_embs, labels = engine.discover_concepts(unlabeled_data)
    
    # 未知のクエリを入力してみる
    test_queries = [
        "How to make a pepperoni pizza?",
        "Watching the football world cup final.",
        "SpaceX successfully landed the booster."
    ]
    
    print("\n==========================================")
    print(" Phase 3.2: Inference on Discovered Valleys")
    print("==========================================")
    
    query_paths = []
    for q in test_queries:
        print(f"\n[Unknown Query]: '{q}'")
        idx, path = engine.run_inference(q)
        print(f"  => Converged to: {engine.domain_texts[idx]}")
        query_paths.append((q, path))
        
    print("\nVisualizing Autonomous Landscape...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # 全データポイント（砂粒）＋発見された谷＋軌跡をあわせてPCA
    all_pts_list = [raw_embs, engine.anchors.numpy()]
    for _, path in query_paths:
        all_pts_list.append(torch.stack(path).numpy())
        
    all_pts = np.concatenate(all_pts_list, axis=0)
    pca.fit(all_pts)
    
    pts_2d = pca.transform(raw_embs)
    anchors_2d = pca.transform(engine.anchors.numpy())
    
    plt.figure(figsize=(12, 9))
    
    # 砂粒（入力データ）のプロット
    scatter = plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6, label='Raw Unlabeled Data (Sand)')
    
    # 発見された谷のプロット
    for i, pt in enumerate(anchors_2d):
        plt.scatter(pt[0], pt[1], color='red', s=600, marker='*', edgecolor='black', zorder=5)
        plt.text(pt[0], pt[1]-0.05, f"Valley #{i}", fontsize=14, weight='bold', color='red', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
    # クエリの軌跡プロット
    colors = ['magenta', 'cyan', 'orange']
    for i, (q, path) in enumerate(query_paths):
        p_2d = pca.transform(torch.stack(path).numpy())
        plt.plot(p_2d[:, 0], p_2d[:, 1], color=colors[i], linestyle='-', linewidth=3, zorder=4)
        plt.scatter(p_2d[0, 0], p_2d[0, 1], color=colors[i], s=200, marker='P', edgecolor='black', label=f"Query: {q.split(' ')[0]}...", zorder=6)
        
    plt.title("Autonomous Concept Discovery (AI building its own landscape)", fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "autonomous_landscape.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to {out_path}")

if __name__ == "__main__":
    run_autonomous_scenario()
