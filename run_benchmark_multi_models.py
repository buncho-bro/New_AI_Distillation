import os
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 1. ベンチマーク用データセット生成
# -----------------------------------------------------------------
CATEGORIES = ['Technology', 'Space', 'Politics', 'Sports']

def generate_benchmark_data(n_samples_per_cat=250):
    templates = {
        'Technology': [
            "The new smartphone features a powerful processor.",
            "Artificial intelligence is transforming the software industry.",
            "Cloud computing allows scalable data storage.",
            "Hackers breached the cybersecurity defenses of the company."
        ],
        'Space': [
            "The Mars rover collected soil samples from the crater.",
            "Astronomers discovered a new exoplanet in the habitable zone.",
            "The rocket launch was delayed due to bad weather.",
            "Astronauts conducted a spacewalk to repair the satellite."
        ],
        'Politics': [
            "The president signed the new economic reform bill.",
            "Senators debated the healthcare policy in the parliament.",
            "The election results showed a significant shift in voting.",
            "Diplomats met to discuss the international trade agreement."
        ],
        'Sports': [
            "The striker scored a hat-trick in the final match.",
            "The tennis champion won her fifth consecutive grand slam.",
            "The basketball team secured their spot in the playoffs.",
            "The Olympic runner broke the world record in the 100m dash."
        ]
    }
    
    clean_data = []
    labels = []
    
    for i, cat in enumerate(CATEGORIES):
        for _ in range(n_samples_per_cat):
            base_text = random.choice(templates[cat])
            clean_data.append(base_text)
            labels.append(i)
            
    return clean_data, labels

def add_noise_to_text(text, noise_prob=0.15):
    """人工的なタイポ（文字の削除、置換）を加えてノイズをシミュレート"""
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and random.random() < noise_prob:
            # 50%で削除、50%でランダムな文字に置換
            if random.random() < 0.5:
                chars[i] = ''
            else:
                chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return "".join(chars)

# -----------------------------------------------------------------
# 2. 物理エンジン (Top-1 Gravity + Harmonic Oscillator)
# -----------------------------------------------------------------
class BenchmarkConceptCore:
    def __init__(self, anchors, dim, steps=100, lr=0.8, momentum=0.4):
        self.anchors = anchors # (N, dim)
        self.dim = dim
        self.steps = steps
        self.lr = lr
        self.momentum = momentum

    def run_physics(self, x_init):
        # x_init: (dim,)
        x = x_init.clone().detach().requires_grad_(True)
        v = torch.zeros_like(x)
        
        for _ in range(self.steps):
            # Top-1 Gravity (一番近い谷を見つける)
            dists = torch.norm(self.anchors - x.unsqueeze(0), dim=1)
            top1_idx = torch.argmin(dists)
            
            # バネのポテンシャルエネルギー (H = 0.5 * k * d^2)
            dist_sq = dists[top1_idx]**2
            H = 0.5 * 1.0 * dist_sq
            
            H.backward()
            grad = x.grad.detach()
            x.grad.zero_()
            
            v = self.momentum * v - self.lr * grad
            
            x_new = x.detach() + v
            x_new = x_new / x_new.norm() # 球面束縛
            x = x_new.requires_grad_(True)
            
            if v.norm() < 1e-4:
                break
                
        final_pos = x.detach()
        dists = torch.norm(self.anchors - final_pos.unsqueeze(0), dim=1)
        return torch.argmin(dists).item()

# -----------------------------------------------------------------
# 3. ベンチマーク実行
# -----------------------------------------------------------------
def run_benchmark():
    models_to_test = [
        {"name": "all-MiniLM-L6-v2", "dim": 384, "desc": "Standard (384d)"},
        {"name": "BAAI/bge-small-en-v1.5", "dim": 384, "desc": "State-of-the-art (384d)"},
        {"name": "all-mpnet-base-v2", "dim": 768, "desc": "High-Dim (768d)"}
    ]
    
    print("Generating Benchmark Dataset...")
    clean_texts, true_labels = generate_benchmark_data(n_samples_per_cat=100) # 400 samples
    noisy_texts = [add_noise_to_text(t, noise_prob=0.20) for t in clean_texts] # 20% typos
    
    # アンカーテキスト（カテゴリの重心となる言葉）
    anchor_texts = [
        "Technology, Software, Computer, AI",
        "Space, Universe, NASA, Planet",
        "Politics, Government, President, Election",
        "Sports, Game, Match, Athlete"
    ]
    
    results = {}
    
    for m_info in models_to_test:
        print(f"\n==========================================")
        print(f" Loading Model: {m_info['name']}")
        print(f"==========================================")
        encoder = SentenceTransformer(m_info['name'])
        
        # アンカーのエンコード
        anchors_emb = torch.tensor(encoder.encode(anchor_texts)).float()
        anchors_emb = anchors_emb / anchors_emb.norm(dim=1, keepdim=True)
        
        # 物理エンジンの初期化
        engine = BenchmarkConceptCore(anchors_emb, dim=m_info['dim'])
        
        # テスト実行関数
        def evaluate(texts):
            embs = torch.tensor(encoder.encode(texts)).float()
            embs = embs / embs.norm(dim=1, keepdim=True)
            
            baseline_correct = 0
            physics_correct = 0
            
            for i in range(len(texts)):
                x = embs[i]
                true_y = true_labels[i]
                
                # Baseline (Cosine Similarity = 最も距離が近いもの)
                dists = torch.norm(anchors_emb - x.unsqueeze(0), dim=1)
                pred_base = torch.argmin(dists).item()
                if pred_base == true_y:
                    baseline_correct += 1
                    
                # Concept Core (Physics = 転がった先の谷)
                pred_phys = engine.run_physics(x)
                if pred_phys == true_y:
                    physics_correct += 1
                    
            acc_base = baseline_correct / len(texts)
            acc_phys = physics_correct / len(texts)
            return acc_base, acc_phys
            
        print("Evaluating on Clean Data...")
        clean_acc_base, clean_acc_phys = evaluate(clean_texts)
        print(f"  [Baseline] Accuracy: {clean_acc_base:.4f}")
        print(f"  [Concept Core] Accuracy: {clean_acc_phys:.4f}")
        
        print("Evaluating on Noisy Data (20% Typos)...")
        noisy_acc_base, noisy_acc_phys = evaluate(noisy_texts)
        print(f"  [Baseline] Accuracy: {noisy_acc_base:.4f}")
        print(f"  [Concept Core] Accuracy: {noisy_acc_phys:.4f}")
        
        results[m_info['desc']] = {
            'clean_base': clean_acc_base,
            'clean_phys': clean_acc_phys,
            'noisy_base': noisy_acc_base,
            'noisy_phys': noisy_acc_phys
        }
        
    # -----------------------------------------------------------------
    # 4. 結果のグラフ描画
    # -----------------------------------------------------------------
    print("\nGenerating Benchmark Chart...")
    labels = list(results.keys())
    
    noisy_base = [results[m]['noisy_base'] * 100 for m in labels]
    noisy_phys = [results[m]['noisy_phys'] * 100 for m in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, noisy_base, width, label='Baseline (Cosine Sim)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, noisy_phys, width, label='Concept Core (Physics)', color='#dd8452')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Robustness Benchmark under 20% Noise (Typos & Errors)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    
    ax.set_ylim(0, 100)
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, weight='bold')
                        
    plt.grid(axis='y', alpha=0.3)
    out_path = os.path.join(OUTPUT_DIR, "benchmark_models.png")
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")

if __name__ == "__main__":
    run_benchmark()
