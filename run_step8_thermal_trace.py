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

class ThermalConceptEngine:
    def __init__(self, anchors, anchor_texts, encoder, dim=384):
        self.encoder = encoder
        self.dim = dim
        self.anchor_texts = anchor_texts
        
        # 基本の物理パラメータ (谷は狭くして概念を明確に分離)
        p = HDParams(dim=dim, use_ridge=False, sigma=0.3, sigma_rep=1.0, alpha_rep=0.5, marble_lr=0.5, marble_steps=200, marble_max_v=1.0, marble_momentum=0.2)
        self.core = HighDimConceptCore(p)
        self.core._centers = anchors.clone()
        self.core._weights = torch.ones(len(anchors)) * 0.5
        
        # 余熱(Thermal Trace)パラメータ
        self.thermal_traces = [] # list of dict: {'pos': tensor, 'heat': float}
        self.heat_decay = 0.8    # 1ターンごとの減衰率 (忘却)
        self.heat_sigma = 5.0    # 余熱が及ぶ広さ (ベースの谷0.3より広く、周辺全体を温める)
        self.initial_heat = 100.0  # 到達した場所に残る初期熱量

    def encode_text(self, text):
        emb = self.encoder.encode([text])
        emb_t = torch.tensor(emb).float()
        return emb_t / emb_t.norm(dim=1, keepdim=True)

    def chat_turn(self, user_text, turn_idx):
        # ユーザー入力をベクトル化
        user_target = self.encode_text(user_text).squeeze(0) # (D,)
        
        # ビー玉をユーザー入力地点に投下
        x = user_target.clone().requires_grad_(True)
        v = torch.zeros_like(x)
        
        # ターン開始時に既存の余熱を減衰 (放熱)
        for trace in self.thermal_traces:
            trace['heat'] *= self.heat_decay
            
        path = [x.detach().clone()]
        
        for step in range(self.core.p.marble_steps):
            # 1. 基本ポテンシャル (静的な谷)
            H_base = self.core.potential(x.unsqueeze(0)).squeeze()
            
            # 2. 余熱ポテンシャル (Thermal Potential)
            H_thermal = torch.tensor(0.0).to(x.device)
            for trace in self.thermal_traces:
                dist_sq = torch.sum((x - trace['pos'])**2)
                # ガウス分布型の余熱ポテンシャル (負の引力)
                H_thermal = H_thermal - trace['heat'] * torch.exp(-dist_sq / (2 * self.heat_sigma**2))
                
            H_total = H_base + H_thermal
            H_total.backward()
            grad = x.grad.detach()
            x.grad.zero_()
            
            # 3. 運動更新
            v = self.core.p.marble_momentum * v - self.core.p.marble_lr * grad
            
            # 速度制限
            v_norm = v.norm()
            if v_norm > self.core.p.marble_max_v:
                v = v * (self.core.p.marble_max_v / v_norm)
                
            x_new = x.detach() + v
            x_new = x_new / x_new.norm() # 球面制約
            
            x = x_new.requires_grad_(True)
            if step % 5 == 0:
                path.append(x.detach().clone())
                
        final_pos = x.detach()
        
        # 到達した場所に新たな「余熱」を残す
        self.thermal_traces.append({'pos': final_pos.clone(), 'heat': self.initial_heat})
        
        # 最も近い概念を出力
        dists = torch.norm(self.core._centers - final_pos.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(dists).item()
        concept = self.anchor_texts[nearest_idx]
        
        return concept, path

def run_thermal_simulation():
    set_seed(42)
    print("Initializing Thermal Concept Engine...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # "Bank" の多義性を判定するためのアンカー群
    anchor_texts = [
        "[Finance] Money, bank account, and finance",
        "[Finance] ATM and credit card",
        "[Nature] River bank, water, and nature",
        "[Nature] Trees, forest, and rocks"
    ]
    
    raw_emb = encoder.encode(anchor_texts)
    DIM = 384
    emb_t = torch.tensor(raw_emb).float()
    emb_t = emb_t / emb_t.norm(dim=1, keepdim=True)
    
    ambiguous_input = "Bank"
    
    def run_scenario(name, context_input):
        print(f"\n==========================================")
        print(f" SCENARIO: {name} Context")
        print(f"==========================================")
        engine = ThermalConceptEngine(emb_t, anchor_texts, encoder, dim=DIM)
        
        # Turn 1: 文脈の生成 (余熱を作る)
        print(f"Turn 1 (Context) [User] : {context_input}")
        concept_1, path_1 = engine.chat_turn(context_input, turn_idx=1)
        print(f"Turn 1 (Concept) [Agent]: {concept_1}")
        print(f"  -> The valley of '{concept_1}' is now heated up.\n")
        
        # Turn 2: 曖昧な入力 (同じ入力文)
        print(f"Turn 2 (Target)  [User] : {ambiguous_input}")
        concept_2, path_2 = engine.chat_turn(ambiguous_input, turn_idx=2)
        print(f"Turn 2 (Concept) [Agent]: => Resolved as: {concept_2}")
        print(f"==========================================\n")
        
        return path_1, path_2
        
    # シナリオ1: 金融の文脈
    f_path1, f_path2 = run_scenario("Finance", "I need to deposit some cash.")
    
    # シナリオ2: 自然の文脈
    t_path1, t_path2 = run_scenario("Nature", "I love fishing in the stream.")
    
    # 可視化 (2つのシナリオのTurn 2の軌跡を比較)
    plot_thermal_comparison(emb_t, anchor_texts, encoder, ambiguous_input, f_path2, t_path2)

def plot_thermal_comparison(anchors, texts, encoder, ambiguous_input, path_food, path_tech):
    ambiguous_emb = encoder.encode([ambiguous_input])
    ambiguous_t = torch.tensor(ambiguous_emb).float()
    ambiguous_t = ambiguous_t / ambiguous_t.norm(dim=1, keepdim=True)
    
    all_pts = anchors.numpy().tolist()
    all_pts.append(ambiguous_t.squeeze().numpy())
    all_pts.extend([p.numpy() for p in path_food])
    all_pts.extend([p.numpy() for p in path_tech])
    
    all_pts = np.array(all_pts)
    vis_pca = PCA(n_components=2)
    pts_2d = vis_pca.fit_transform(all_pts)
    
    anchors_2d = pts_2d[:len(anchors)]
    ambig_2d = pts_2d[len(anchors)]
    
    plt.figure(figsize=(12, 8))
    
    # アンカーのプロット
    for i, (pt, txt) in enumerate(zip(anchors_2d, texts)):
        color = 'red' if 'Finance' in txt else 'blue'
        marker = 'o' if 'Finance' in txt else 's'
        plt.scatter(pt[0], pt[1], marker=marker, s=300, color=color, edgecolor='black', alpha=0.8, zorder=5)
        plt.text(pt[0], pt[1]-0.05, txt, fontsize=10, weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
    # 曖昧な入力のプロット
    plt.scatter(ambig_2d[0], ambig_2d[1], marker='*', s=400, color='gold', edgecolor='black', zorder=6, label="Ambiguous Input:\n'Bank'")
    
    # Finance Context 軌跡
    pf_2d = pts_2d[len(anchors)+1 : len(anchors)+1+len(path_food)]
    plt.plot(pf_2d[:, 0], pf_2d[:, 1], color='red', linewidth=3, linestyle='-', zorder=3, label="Trajectory (Finance Context)")
    plt.scatter(pf_2d[-1, 0], pf_2d[-1, 1], color='red', s=100, marker='X', edgecolors='black', zorder=6)
    
    # Nature Context 軌跡
    pt_2d = pts_2d[len(anchors)+1+len(path_food) :]
    plt.plot(pt_2d[:, 0], pt_2d[:, 1], color='blue', linewidth=3, linestyle='-', zorder=3, label="Trajectory (Nature Context)")
    plt.scatter(pt_2d[-1, 0], pt_2d[-1, 1], color='blue', s=100, marker='X', edgecolors='black', zorder=6)
    
    plt.title("Thermal Trace (Contextual Gravity) Resolving Ambiguity")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "thermal_context.png"))
    print(f"Trajectory comparison saved to {os.path.join(OUTPUT_DIR, 'thermal_context.png')}")

if __name__ == "__main__":
    run_thermal_simulation()
