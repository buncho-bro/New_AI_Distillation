import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# 概念核モジュール（既存のものを拡張して利用）
from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class StatefulConceptAgent:
    def __init__(self, anchors, anchor_texts, encoder, pca, dim=8):
        self.encoder = encoder
        self.pca = pca
        self.dim = dim
        self.anchor_texts = anchor_texts
        
        # 物理パラメータ (遷移しやすさを持たせるため、少し引力範囲を広めにする)
        p = HDParams(dim=dim, use_ridge=False, sigma=0.3, sigma_rep=1.2, alpha_rep=0.5, marble_lr=0.5, marble_steps=100, marble_max_v=0.5)
        self.core = HighDimConceptCore(p)
        self.core._centers = anchors.clone()
        self.core._weights = torch.ones(len(anchors)) * 0.5
        
        # 状態 (ビー玉の現在地) - 初回入力時に設定するため最初は None
        self.current_pos = None
        
        # 軌跡保存用
        self.trajectory = []
        self.states_log = []

    def encode_text(self, text):
        emb = self.encoder.encode([text])
        emb_pca = torch.tensor(emb).float()
        return emb_pca / emb_pca.norm(dim=1, keepdim=True)

    def chat_turn(self, user_text, kick_strength=1.5):
        # 1. ユーザー入力をベクトル化
        user_target = self.encode_text(user_text).squeeze(0) # (D,)
        
        # Debug: Check nearest anchor to user_target
        dists_target = torch.norm(self.core._centers - user_target.unsqueeze(0), dim=1)
        print(f"    (Debug) Nearest to User Input: {self.anchor_texts[torch.argmin(dists_target).item()]}")
        
        if self.current_pos is None:
            self.current_pos = user_target.clone()
            self.trajectory = [self.current_pos.clone()]
            
        # 2. 文脈(10%)と新規入力(90%)をブレンドして新しい思考の起点とする
        x = (self.current_pos.detach() * 0.1 + user_target * 0.9)
        x = x / x.norm()
        
        # Debug: Check nearest anchor to blended start
        dists_start = torch.norm(self.core._centers - x.unsqueeze(0), dim=1)
        print(f"    (Debug) Nearest to Blended Start: {self.anchor_texts[torch.argmin(dists_start).item()]}")
        
        x = x.requires_grad_(True)
        v = torch.zeros_like(x)
        
        path = [x.detach().clone()]
        
        for _ in range(self.core.p.marble_steps):
            # 基本ポテンシャル (谷と尾根)
            H_base = self.core.potential(x.unsqueeze(0)).squeeze()
            H_base.backward()
            grad = x.grad.detach()
            
            # 運動更新
            v = self.core.p.marble_momentum * v - self.core.p.marble_lr * grad
            v_norm = v.norm()
            if v_norm > self.core.p.marble_max_v:
                v = v * (self.core.p.marble_max_v / v_norm)
                
            x_new = x.detach() + v
            x_new = x_new / x_new.norm() # 球面制約
            
            x = x_new.requires_grad_(True)
            path.append(x.detach().clone())
            
            if v.norm() < self.core.p.marble_vel_thresh:
                break
                
        # 更新
        self.current_pos = x.detach()
        self.trajectory.append(self.current_pos.clone())
        
        # Debug: Check nearest anchor after physics
        dists_after = torch.norm(self.core._centers - self.current_pos.unsqueeze(0), dim=1)
        print(f"    (Debug) Nearest After Physics: {self.anchor_texts[torch.argmin(dists_after).item()]}")
        
        # デコード (最も近いアンカー概念を出力)
        dists = torch.norm(self.core._centers - self.current_pos.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(dists).item()
        nearest_dist = dists[nearest_idx].item()
        
        response = self.anchor_texts[nearest_idx]
        self.states_log.append((user_text, response, path))
        
        return response

def run_stateful_simulation():
    set_seed(42)
    print("Initializing Stateful Concept Agent...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # ITサポートドメインの概念アンカー（谷）を定義
    anchor_texts = [
        "Greeting: Hello! How can I help you today?",
        "Ask Context: Please provide more details about the error.",
        "Fix Network: It seems like a network issue. Please check your internet connection and DNS.",
        "Fix Database: This is a database connection error. Check your DB credentials.",
        "Fix Syntax: You have a syntax or typo in your code. Please review the lines.",
        "Resolution: Great! I'm glad the problem is solved."
    ]
    
    raw_emb = encoder.encode(anchor_texts)
    DIM = 384
    pca_emb = torch.tensor(raw_emb).float()
    pca_emb = pca_emb / pca_emb.norm(dim=1, keepdim=True)
    
    agent = StatefulConceptAgent(pca_emb, anchor_texts, encoder, None, dim=DIM)
    
    # ユーザーとの疑似対話シナリオ
    scenario = [
        "Hi there!",
        "My python backend crashed suddenly.",
        "The log says 'connection refused' on port 5432.",
        "Ah, I had the wrong password in my env file. Fixed it and it works now!"
    ]
    
    print("\n--- STARTING CHAT SCENARIO ---")
    for msg in scenario:
        print(f"\n[User]: {msg}")
        response = agent.chat_turn(msg)
        print(f"[Agent (Concept)]: {response}")
        
    # 可視化 (2D PCAで軌跡をプロット)
    print("\nVisualizing Trajectory...")
    plot_trajectory(agent, pca_emb, anchor_texts)

def plot_trajectory(agent, anchors, texts):
    all_pts = anchors.numpy().tolist()
    
    for pt in agent.trajectory:
        all_pts.append(pt.numpy())
        
    all_pts = np.array(all_pts)
    vis_pca = PCA(n_components=2)
    pts_2d = vis_pca.fit_transform(all_pts)
    
    anchors_2d = pts_2d[:len(anchors)]
    traj_2d = pts_2d[len(anchors):]
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(anchors)))
    for i, (pt, txt) in enumerate(zip(anchors_2d, texts)):
        label_short = txt.split(":")[0]
        plt.scatter(pt[0], pt[1], marker='*', s=300, color=colors[i], label=label_short, edgecolor='black', zorder=5)
        plt.text(pt[0]+0.02, pt[1]+0.02, label_short, fontsize=10, weight='bold')
        
    for i in range(len(traj_2d) - 1):
        start = traj_2d[i]
        end = traj_2d[i+1]
        plt.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8, alpha=0.5)
        )
        plt.text(start[0]-0.05, start[1]+0.05, f"T{i}", color='red', fontsize=14, weight='bold', zorder=10)
    
    plt.text(traj_2d[-1][0]-0.05, traj_2d[-1][1]+0.05, f"T{len(traj_2d)-1}", color='red', fontsize=14, weight='bold', zorder=10)
    plt.scatter(traj_2d[:, 0], traj_2d[:, 1], c='red', s=100, zorder=6, label="Context Marble")
        
    plt.title("Stateful Concept Agent: Thought Trajectory")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stateful_trajectory.png"))
    print(f"Trajectory saved to {os.path.join(OUTPUT_DIR, 'stateful_trajectory.png')}")

if __name__ == "__main__":
    run_stateful_simulation()
