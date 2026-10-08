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

class AudioDenoisingEngine:
    def __init__(self, encoder, dim=384):
        self.encoder = encoder
        self.dim = dim
        
        # 音声認識のデノイズを検証するためのアンカー（谷）
        self.domain_texts = [
            "[Tech] Mobile App, Download Application, Software",
            "[Food] Fresh Apple, Buy Fruits, Grocery",
            "[Nature] Wild Ape, Monkey, Gorilla, Animal",
            "[Travel] North Pole, South Pole, Cold Geography"
        ]
        
        raw_emb = self.encoder.encode(self.domain_texts)
        self.anchors = torch.tensor(raw_emb).float()
        self.anchors = self.anchors / self.anchors.norm(dim=1, keepdim=True)
        
        # 物理パラメータ (デノイズ効果を高めるため、引力を広く強くする)
        p = HDParams(
            dim=self.dim, use_ridge=False,
            sigma=0.5, sigma_rep=1.0, alpha_rep=0.5,
            marble_lr=0.5, marble_steps=150,
            marble_max_v=1.0, marble_momentum=0.4
        )
        self.core = HighDimConceptCore(p)
        self.core._centers = self.anchors.clone()
        self.core._weights = torch.ones(len(self.anchors)) * 1.5 # 引力を強くする
        
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
        return nearest_idx, path

def run_audio_denoising_scenario():
    print("Initializing Audio Denoising Engine...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    engine = AudioDenoisingEngine(encoder, dim=384)
    
    # テストシナリオ
    scenarios = [
        {
            "intent": "App",
            "perfect": "I want to download an app.",
            "whisper_error": "I want to down load an ape."
        },
        {
            "intent": "Apple",
            "perfect": "I want to buy an apple.",
            "whisper_error": "I want to buy an app pole."
        }
    ]
    
    results_paths = {}
    
    for s in scenarios:
        print(f"\n==========================================")
        print(f" SCENARIO: Intended meaning = {s['intent']}")
        print(f"==========================================")
        
        # 完璧な音声テキスト
        print(f"[Perfect Audio]: '{s['perfect']}'")
        idx_p, path_p = engine.run_inference(s['perfect'])
        print(f"  => Converged to: {engine.domain_texts[idx_p]}")
        
        # 誤認識された音声テキスト
        print(f"\n[Whisper Error]: '{s['whisper_error']}'")
        idx_e, path_e = engine.run_inference(s['whisper_error'])
        print(f"  => Converged to: {engine.domain_texts[idx_e]}")
        
        if idx_p == idx_e:
            print("  [SUCCESS] Physical Denoising absorbed the transcription error!")
        else:
            print("  [FAILED] Transcription error broke the inference.")
            
        results_paths[s['intent']] = {
            'perfect_path': path_p,
            'error_path': path_e,
            'perfect_text': s['perfect'],
            'error_text': s['whisper_error']
        }
        
    # 可視化 (AppのシナリオをPCA投影)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    s_app = results_paths["App"]
    all_pts = torch.cat([
        engine.anchors, 
        s_app['perfect_path'][0].unsqueeze(0),
        s_app['error_path'][0].unsqueeze(0)
    ]).numpy()
    
    pca.fit(all_pts)
    
    anchors_2d = pca.transform(engine.anchors.numpy())
    path_p_2d = pca.transform(torch.stack(s_app['perfect_path']).numpy())
    path_e_2d = pca.transform(torch.stack(s_app['error_path']).numpy())
    
    plt.figure(figsize=(10, 8))
    
    # アンカー(谷)
    colors = ['blue', 'red', 'green', 'cyan']
    for i, (pt, txt) in enumerate(zip(anchors_2d, engine.domain_texts)):
        plt.scatter(pt[0], pt[1], color=colors[i], s=400, marker='*', edgecolor='black', zorder=5)
        plt.text(pt[0], pt[1]-0.05, txt.split(']')[0]+']', fontsize=12, weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
    # 軌跡
    plt.plot(path_p_2d[:, 0], path_p_2d[:, 1], color='blue', linestyle='-', linewidth=2, label='Path (Perfect Audio)', zorder=3)
    plt.plot(path_e_2d[:, 0], path_e_2d[:, 1], color='red', linestyle='--', linewidth=3, label='Path (Whisper Error)', zorder=4)
    
    # スタート地点
    plt.scatter(path_p_2d[0, 0], path_p_2d[0, 1], color='black', s=100, label=f"Start: '{s_app['perfect_text']}'", zorder=6)
    plt.scatter(path_e_2d[0, 0], path_e_2d[0, 1], color='orange', s=150, marker='X', edgecolor='black', label=f"Start: '{s_app['error_text']}'", zorder=6)
    
    plt.title("Audio Transcription Denoising via Concept Core", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "audio_denoising.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to {out_path}")

if __name__ == "__main__":
    run_audio_denoising_scenario()
