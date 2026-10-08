import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class LearningConceptEngine:
    def __init__(self, encoder, dim=384):
        self.encoder = encoder
        self.dim = dim
        
        # 4つのアンカー（知識ドメイン）
        self.domain_texts = [
            "[Food] Apple, Banana, Orange, Fruits",
            "[Tech] Apple MacBook, iPhone, PC",
            "[Finance] Bank, Money, Cash",
            "[Nature] River Bank, Water, Tree"
        ]
        
        raw_emb = self.encoder.encode(self.domain_texts)
        self.anchors = torch.tensor(raw_emb).float()
        self.anchors = self.anchors / self.anchors.norm(dim=1, keepdim=True)
        
        # 物理パラメータ (反発力 use_ridge=True)
        p = HDParams(
            dim=self.dim, use_ridge=True,
            sigma=0.3, sigma_rep=0.4, alpha_rep=0.5,
            marble_lr=0.5, marble_steps=100,
            marble_max_v=1.0, marble_momentum=0.2
        )
        self.core = HighDimConceptCore(p)
        self.core._centers = self.anchors.clone()
        self.core._weights = torch.ones(len(self.anchors)) * 0.5
        self.core._ridges = [] # 動的に追加される反発点(尾根)
        
    def encode_text(self, text):
        emb = self.encoder.encode([text])
        vec = torch.tensor(emb).float().squeeze(0)
        return vec / vec.norm()

    def run_inference(self, user_text):
        x = self.encode_text(user_text).clone().requires_grad_(True)
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
        return nearest_idx, final_pos, path

    def add_negative_feedback(self, anchor_idx, penalty=0.5):
        """不正解だった谷の引力を弱める（埋め立てる）"""
        self.core._weights[anchor_idx] = max(0.0, self.core._weights[anchor_idx] - penalty)
        print(f"  [Learning] Weakened the valley of '{self.domain_texts[anchor_idx]}' by -{penalty}!")

    def add_positive_feedback(self, anchor_idx, query_pos, shift=0.5):
        """正解だった谷を、入力クエリの方向に移動させる（引力の重心移動）"""
        self.core._centers[anchor_idx] = (1 - shift) * self.core._centers[anchor_idx] + shift * query_pos
        self.core._centers[anchor_idx] /= self.core._centers[anchor_idx].norm()
        # ついでに深さも増やす
        self.core._weights[anchor_idx] += 1.0
        print(f"  [Learning] Shifted and Deepened the valley of '{self.domain_texts[anchor_idx]}' towards the query!")

def run_learning_scenario():
    print("Initializing Learning Engine...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    engine = LearningConceptEngine(encoder, dim=384)
    
    # 曖昧なクエリ: "Apple" (デフォルトではTechに落ちやすい)
    query = "Apple"
    
    print("\n==========================================")
    print(" Turn 1: Initial Inference (No Learning)")
    print("==========================================")
    print(f"[User] Query: '{query}'")
    idx_1, pos_1, path_1 = engine.run_inference(query)
    print(f"[Agent] Converged to: {engine.domain_texts[idx_1]}")
    
    print("\n==========================================")
    print(" Feedback: Negative (That's wrong!)")
    print("==========================================")
    # ユーザーが「いや、果物のことだよ！」とフィードバックしたと仮定
    print("[User] No, I meant the fruit!")
    # エージェントは「落ちた谷(Tech)」が間違いだったと学習し、その谷を埋め立てる
    engine.add_negative_feedback(1, penalty=0.5)
    # 正解の谷(Food)をクエリの方向へ大きく移動させる
    query_pos = engine.encode_text(query)
    engine.add_positive_feedback(0, query_pos, shift=0.6)
    
    print("\n==========================================")
    print(" Turn 2: Inference After Learning")
    print("==========================================")
    print(f"[User] Query: '{query}' (Again)")
    idx_2, pos_2, path_2 = engine.run_inference(query)
    print(f"[Agent] Converged to: {engine.domain_texts[idx_2]}")
    if idx_2 == 0:
        print("  => SUCCESS! The correct valley shifted to capture the concept.")
        
    # 可視化 (PCAで2Dに投影)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    # 全てのポイントを集めてPCAをfitさせる
    all_pts = torch.cat([engine.anchors, pos_1.unsqueeze(0), pos_2.unsqueeze(0)]).numpy()
    pca.fit(all_pts)
    
    anchors_2d = pca.transform(engine.anchors.numpy())
    path1_2d = pca.transform(torch.stack(path_1).numpy())
    path2_2d = pca.transform(torch.stack(path_2).numpy())
    
    plt.figure(figsize=(10, 8))
    
    # アンカー(谷)
    colors = ['red', 'blue', 'green', 'purple']
    for i, (pt, txt) in enumerate(zip(anchors_2d, engine.domain_texts)):
        plt.scatter(pt[0], pt[1], color=colors[i], s=300, marker='*', edgecolor='black', zorder=5)
        plt.text(pt[0], pt[1]-0.05, txt.split(']')[0]+']', fontsize=12, weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
    # 軌跡
    plt.plot(path1_2d[:, 0], path1_2d[:, 1], color='gray', linestyle='--', linewidth=2, label='Turn 1 (Wrong Path)', zorder=2)
    plt.plot(path2_2d[:, 0], path2_2d[:, 1], color='blue', linestyle='-', linewidth=3, label='Turn 2 (Correct Path after Learning)', zorder=3)
    
    # スタート地点
    start_pt = path1_2d[0]
    plt.scatter(start_pt[0], start_pt[1], color='black', s=100, label='Query: "Apple"', zorder=6)
    
    plt.title("Reinforcement Landscaping (Learning via Valley Depth Adjustment)", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "learning_landscape.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to {out_path}")

if __name__ == "__main__":
    # HighDimConceptCore を少しモンキーパッチして、_ridges リストを使えるようにする
    # 元の実装に _ridges がない場合のために安全に拡張
    original_potential = HighDimConceptCore.potential
    
    def potential_with_ridges(self, x):
        H = original_potential(self, x)
        if hasattr(self, '_ridges') and len(self._ridges) > 0:
            for ridge in self._ridges:
                r_pos = ridge['pos'].unsqueeze(0)
                alpha = ridge['alpha']
                dist_sq = torch.sum((x - r_pos)**2, dim=1)
                H += alpha * torch.exp(-dist_sq / (2 * self.p.sigma_rep**2))
        return H
        
    HighDimConceptCore.potential = potential_with_ridges
    
    run_learning_scenario()
