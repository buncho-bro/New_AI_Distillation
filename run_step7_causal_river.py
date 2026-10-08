import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class CausalConceptEngine:
    def __init__(self, anchors, anchor_texts, encoder, pca, dim=384):
        self.encoder = encoder
        self.pca = pca
        self.dim = dim
        self.anchor_texts = anchor_texts
        
        # 物理パラメータ (谷を少し狭くし、因果の風で押し出せるようにする)
        p = HDParams(dim=dim, use_ridge=False, sigma=0.5, sigma_rep=1.0, alpha_rep=0.5, marble_lr=0.5, marble_steps=400, marble_max_v=1.0, marble_momentum=0.5)
        self.core = HighDimConceptCore(p)
        self.core._centers = anchors.clone()
        self.core._weights = torch.ones(len(anchors)) * 0.5
        
        self.causal_links = [] # (source_idx, target_idx, weight, radius)

    def add_causal_link(self, src_idx, tgt_idx, weight=1.5, radius=0.6):
        """AからBへの因果の風（ベクトル場）を追加"""
        self.causal_links.append((src_idx, tgt_idx, weight, radius))

    def encode_text(self, text):
        emb = self.encoder.encode([text])
        emb_t = torch.tensor(self.pca.transform(emb)).float()
        return emb_t / emb_t.norm(dim=1, keepdim=True)

    def simulate_causal_chain(self, start_text):
        # 初期座標 (ユーザー入力を投下)
        x = self.encode_text(start_text).squeeze(0).requires_grad_(True)
        v = torch.zeros_like(x)
        
        path = [x.detach().clone()]
        active_concepts_log = []
        
        for step in range(self.core.p.marble_steps):
            # 1. 基本ポテンシャル (静的な谷の引力)
            H_base = self.core.potential(x.unsqueeze(0)).squeeze()
            H_base.backward()
            grad_base = x.grad.detach()
            x.grad.zero_()
            
            # 2. 因果ベクトル場 (Causal Wind)
            F_causal = torch.zeros_like(x)
            for src_idx, tgt_idx, weight, radius in self.causal_links:
                src_pos = self.core._centers[src_idx]
                tgt_pos = self.core._centers[tgt_idx]
                
                # 現在地から原因(src)までの距離
                dist_to_src_sq = torch.sum((x.detach() - src_pos)**2)
                
                # 原因の近くにいるときだけ、結果(tgt)へ向かう風が吹く
                activation = torch.exp(-dist_to_src_sq / (2 * radius**2))
                wind_dir = (tgt_pos - x.detach())
                F_causal += weight * activation * wind_dir
            
            # 3. 運動更新 (引力の勾配 + 因果の風)
            # F = -∇H + F_causal
            total_force = -grad_base + F_causal
            
            v = self.core.p.marble_momentum * v + self.core.p.marble_lr * total_force
            
            # 速度制限
            v_norm = v.norm()
            if v_norm > self.core.p.marble_max_v:
                v = v * (self.core.p.marble_max_v / v_norm)
                
            x_new = x.detach() + v
            x_new = x_new / x_new.norm() # 球面制約
            
            x = x_new.requires_grad_(True)
            
            # ログ記録 (少し間引き)
            if step % 5 == 0:
                path.append(x.detach().clone())
                # 現在最も近い概念を記録
                dists = torch.norm(self.core._centers - x.detach().unsqueeze(0), dim=1)
                nearest_idx = torch.argmin(dists).item()
                if dists[nearest_idx] < 0.6: # 谷の近くにいる場合のみ
                    concept = self.anchor_texts[nearest_idx]
                    if len(active_concepts_log) == 0 or active_concepts_log[-1] != concept:
                        active_concepts_log.append(concept)
                        print(f"Step {step:03d} | Reached Concept: [{concept}]")
            
        return path, active_concepts_log

def run_causal_simulation():
    set_seed(42)
    print("Initializing Causal Concept Engine...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 因果連鎖を構成する概念群
    anchor_texts = [
        "Heavy Rain",       # 0
        "Wet Road",         # 1
        "Slippery",         # 2
        "Traffic Jam"       # 3
    ]
    
    raw_emb = encoder.encode(anchor_texts)
    DIM = 4
    pca = PCA(n_components=DIM, whiten=True)
    emb_t = torch.tensor(pca.fit_transform(raw_emb)).float()
    emb_t = emb_t / emb_t.norm(dim=1, keepdim=True)
    
    engine = CausalConceptEngine(emb_t, anchor_texts, encoder, pca, dim=DIM)
    
    # 因果リンクの追加 (A -> B)
    # 重み(weight)と影響半径(radius)を設定
    # radiusを広めにすることで、谷の底からでも風を受けて脱出できるようにする
    engine.add_causal_link(0, 1, weight=3.0, radius=0.6) # Rain -> Wet
    engine.add_causal_link(1, 2, weight=3.0, radius=0.6) # Wet -> Slippery
    engine.add_causal_link(2, 3, weight=3.0, radius=0.6) # Slippery -> Traffic Jam
    
    print("\n--- STARTING CAUSAL INFERENCE SIMULATION ---")
    user_input = "It is pouring outside!"
    print(f"User Input: '{user_input}'\n")
    
    path, concepts_traversed = engine.simulate_causal_chain(user_input)
    
    print(f"\nFinal Inference Chain: {' -> '.join(concepts_traversed)}")
    
    print("\nVisualizing Causal Trajectory...")
    plot_causal_trajectory(engine, emb_t, anchor_texts, path)

def plot_causal_trajectory(engine, anchors, texts, path):
    # PCAで2D化 (アンカーと軌跡)
    all_pts = anchors.numpy().tolist()
    path_pts = [p.numpy() for p in path]
    all_pts.extend(path_pts)
    
    all_pts = np.array(all_pts)
    vis_pca = PCA(n_components=2)
    pts_2d = vis_pca.fit_transform(all_pts)
    
    anchors_2d = pts_2d[:len(anchors)]
    traj_2d = pts_2d[len(anchors):]
    
    plt.figure(figsize=(10, 8))
    
    # 因果リンクの描画 (背景)
    for src_idx, tgt_idx, _, _ in engine.causal_links:
        src_pt = anchors_2d[src_idx]
        tgt_pt = anchors_2d[tgt_idx]
        plt.annotate(
            '', xy=tgt_pt, xytext=src_pt,
            arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0.1, alpha=0.2, width=3, headwidth=12)
        )
    
    # アンカーのプロット
    colors = plt.cm.plasma(np.linspace(0, 1, len(anchors)))
    for i, (pt, txt) in enumerate(zip(anchors_2d, texts)):
        plt.scatter(pt[0], pt[1], marker='s', s=400, color=colors[i], edgecolor='black', zorder=5)
        plt.text(pt[0], pt[1]-0.05, txt, fontsize=12, weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
    # 軌跡のプロット (グラデーションで時間の経過を表現)
    cmap = plt.cm.Greys
    for i in range(len(traj_2d) - 1):
        start = traj_2d[i]
        end = traj_2d[i+1]
        c = cmap(0.3 + 0.7 * (i / len(traj_2d)))
        plt.plot([start[0], end[0]], [start[1], end[1]], color=c, linewidth=2, zorder=3)
        if i % 10 == 0:
            plt.scatter(start[0], start[1], color=c, s=20, zorder=4)
            
    # スタート地点とエンド地点の強調
    plt.scatter(traj_2d[0, 0], traj_2d[0, 1], c='green', s=200, marker='o', edgecolors='black', zorder=6, label='Start (User Input)')
    plt.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='red', s=200, marker='X', edgecolors='black', zorder=6, label='End (Inference)')
        
    plt.title("Causal River: Autonomous Inference Trajectory in Concept Core")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "causal_trajectory.png"))
    print(f"Trajectory saved to {os.path.join(OUTPUT_DIR, 'causal_trajectory.png')}")

if __name__ == "__main__":
    run_causal_simulation()
