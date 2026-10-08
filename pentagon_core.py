import os
import torch
from sentence_transformers import SentenceTransformer

class PentagonEngine:
    """
    Pentagonの本格活用向けコアエンジン。
    Top-1 Gravity (バネポテンシャル)、Thermal Trace (文脈の余熱)、
    および Gravitational Shift (リアルタイム学習) を完全に統合。
    """
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2", dim=384):
        self.dim = dim
        self.encoder = SentenceTransformer(model_name)
        
        # 脳の状態（地形）
        self.anchors = None      # (N, dim)
        self.weights = None      # (N,)
        self.domain_names = []   # カテゴリ名のリスト
        
        # 物理パラメータ
        self.lr = 0.8
        self.momentum = 0.4
        self.steps = 100
        
        # 余熱（文脈）メモリ: 最近使われた谷のインデックスと熱量
        self.thermal_memory = {}
        self.thermal_decay = 0.8
        self.thermal_boost = 1.0

    def load_landscape(self, path):
        """ディスクから学習済みの地形（脳）をロード"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Landscape file not found: {path}")
        data = torch.load(path)
        self.anchors = data['anchors']
        self.weights = data['weights']
        self.domain_names = data['domain_names']
        print(f"[Pentagon Core] Loaded landscape with {len(self.anchors)} valleys from {path}.")

    def save_landscape(self, path):
        """現在の地形（脳）をディスクに保存（成長の永続化）"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            'anchors': self.anchors,
            'weights': self.weights,
            'domain_names': self.domain_names
        }
        torch.save(data, path)
        print(f"[Pentagon Core] Saved landscape to {path}.")

    def inference(self, text, use_thermal=True):
        """物理ポテンシャルによる推論（推論結果のインデックスを返す）"""
        if self.anchors is None:
            raise ValueError("Landscape is not loaded. Call load_landscape() first.")
            
        emb = self.encoder.encode([text])
        x = torch.tensor(emb).float().squeeze(0)
        x = x / x.norm()
        x = x.requires_grad_(True)
        
        v = torch.zeros_like(x)
        
        # 有効な重み（ベースの深さ ＋ 余熱）
        effective_weights = self.weights.clone()
        if use_thermal:
            for idx, heat in self.thermal_memory.items():
                effective_weights[idx] += heat * self.thermal_boost
                
        for _ in range(self.steps):
            dists = torch.norm(self.anchors - x.unsqueeze(0), dim=1)
            top1_idx = torch.argmin(dists)
            
            # Top-1 バネポテンシャル (Hooke's Law)
            dist_sq = dists[top1_idx]**2
            H = 0.5 * effective_weights[top1_idx] * dist_sq
            
            H.backward()
            grad = x.grad.detach()
            x.grad.zero_()
            
            v = self.momentum * v - self.lr * grad
            
            x_new = x.detach() + v
            x_new = x_new / x_new.norm()
            x = x_new.requires_grad_(True)
            
            if v.norm() < 1e-4:
                break
                
        final_pos = x.detach()
        dists = torch.norm(self.anchors - final_pos.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(dists).item()
        
        # 決定した谷に余熱を追加し、他の余熱を冷ます
        if use_thermal:
            for k in self.thermal_memory.keys():
                self.thermal_memory[k] *= self.thermal_decay
            self.thermal_memory[nearest_idx] = self.thermal_memory.get(nearest_idx, 0) + 1.0
            
        return nearest_idx, self.domain_names[nearest_idx]

    def learn_from_feedback(self, text, target_idx, shift_ratio=0.1, depth_boost=0.5):
        """
        ユーザーのフィードバックによるリアルタイム強化学習（Gravitational Shift）。
        正しい谷をクエリの方へ少し引っ張り、谷をさらに深くする。
        """
        emb = self.encoder.encode([text])
        query_pos = torch.tensor(emb).float().squeeze(0)
        query_pos = query_pos / query_pos.norm()
        
        # 1. 谷の位置をクエリの方へシフト
        target_anchor = self.anchors[target_idx]
        new_anchor = (1 - shift_ratio) * target_anchor + shift_ratio * query_pos
        new_anchor = new_anchor / new_anchor.norm()
        self.anchors[target_idx] = new_anchor
        
        # 2. 谷をさらに深くする（重力強化）
        self.weights[target_idx] += depth_boost
        
        print(f"[Pentagon Core] Learning applied: Shifted valley '{self.domain_names[target_idx]}' towards query and increased depth to {self.weights[target_idx]:.2f}.")
