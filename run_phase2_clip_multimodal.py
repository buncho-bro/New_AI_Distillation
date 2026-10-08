import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MultimodalConceptEngine:
    def __init__(self, encoder, dim=512):
        self.encoder = encoder
        self.dim = dim
        
        # 概念の谷（アンカー）をテキストで定義
        self.domain_texts = [
            "[Pet] Dog, Puppy, Golden Retriever",
            "[Vehicle] Car, Automobile, Sports car",
            "[Food] Pizza, Delicious Italian food, Slice"
        ]
        
        # テキストをエンコードしてアンカーの座標とする
        raw_emb = self.encoder.encode(self.domain_texts)
        self.anchors = torch.tensor(raw_emb).float()
        self.anchors = self.anchors / self.anchors.norm(dim=1, keepdim=True)
        
        # 物理エンジン
        p = HDParams(
            dim=self.dim, use_ridge=False,
            sigma=0.4, sigma_rep=1.0, alpha_rep=0.5,
            marble_lr=0.5, marble_steps=150,
            marble_max_v=1.0, marble_momentum=0.2
        )
        self.core = HighDimConceptCore(p)
        self.core._centers = self.anchors.clone()
        self.core._weights = torch.ones(len(self.anchors)) * 1.0
        
    def encode_input(self, data):
        """テキストまたは画像をエンコードする"""
        emb = self.encoder.encode([data])
        vec = torch.tensor(emb).float().squeeze(0)
        return vec / vec.norm()

    def run_inference(self, data):
        x = self.encode_input(data).clone().requires_grad_(True)
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

import requests
from io import BytesIO

def download_image(url, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(filepath, 'wb') as f:
        f.write(response.content)
    return Image.open(filepath)

def run_multimodal_scenario():
    print("Loading Multimodal CLIP Engine...")
    # 画像とテキストを512次元の同一空間にマッピングするCLIPモデル
    encoder = SentenceTransformer("clip-ViT-B-32")
    engine = MultimodalConceptEngine(encoder, dim=512)
    
    print("\nDownloading test images...")
    img_dog = download_image("https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400", "test_dog.jpg")
    img_car = download_image("https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=400", "test_car.jpg")
    
    scenarios = [
        {"type": "Image", "data": img_dog, "label": "Photo of a Dog"},
        {"type": "Text",  "data": "A very cute golden retriever.", "label": "Text: 'A very cute golden retriever.'"},
        {"type": "Image", "data": img_car, "label": "Photo of a Car"}
    ]
    
    paths = []
    
    print("\n==========================================")
    print(" Running Multimodal Inference (Text & Image)")
    print("==========================================")
    
    for s in scenarios:
        print(f"\n[Input {s['type']}]: {s['label']}")
        idx, path = engine.run_inference(s['data'])
        print(f"  => Converged to Valley: {engine.domain_texts[idx]}")
        paths.append((s, path))
        
    print("\nVisualizing Multimodal Landscape...")
    # PCAで可視化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # 全てのアンカーと、各シナリオのスタート/エンド地点を結合してPCA
    all_pts_list = [engine.anchors]
    for _, path in paths:
        all_pts_list.append(torch.stack(path))
    all_pts = torch.cat(all_pts_list).numpy()
    
    pca.fit(all_pts)
    anchors_2d = pca.transform(engine.anchors.numpy())
    
    plt.figure(figsize=(10, 8))
    
    # アンカー(谷)のプロット
    colors = ['orange', 'blue', 'green']
    for i, (pt, txt) in enumerate(zip(anchors_2d, engine.domain_texts)):
        plt.scatter(pt[0], pt[1], color=colors[i], s=500, marker='*', edgecolor='black', zorder=5)
        plt.text(pt[0], pt[1]-0.05, txt.split(']')[0]+']', fontsize=12, weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
    # 軌跡のプロット
    path_colors = ['red', 'purple', 'cyan']
    for i, (s, path) in enumerate(paths):
        p_2d = pca.transform(torch.stack(path).numpy())
        plt.plot(p_2d[:, 0], p_2d[:, 1], color=path_colors[i], linestyle='-', linewidth=2, zorder=3)
        # スタート地点
        marker = 's' if s['type'] == 'Image' else 'o'
        plt.scatter(p_2d[0, 0], p_2d[0, 1], color=path_colors[i], s=150, marker=marker, edgecolor='black', label=f"Input: {s['label']}", zorder=6)
        
    plt.title("Multimodal Concept Core (Images & Text falling into the same valleys)", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "multimodal_landscape.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to {out_path}")

if __name__ == "__main__":
    run_multimodal_scenario()
