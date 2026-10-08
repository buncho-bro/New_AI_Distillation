"""
Pentagon - Physics-Based Concept Core Chat AI
=============================================
概念核（物理的ポテンシャル＋余熱モデル）を搭載した対話型AIシステム。
ビー玉が高次元空間を転がり、文脈に沿った「谷」へ収束することで、
ブレない推論と文脈の統一を実現します。

起動方法: streamlit run app_pentagon.py
"""

import os
import sys
import torch
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# 概念核エンジンのインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from high_dim_concept_core import HighDimConceptCore, HDParams

# ========== 知識ドメイン定義 ==========
KNOWLEDGE_DOMAINS = {
    "greeting": {
        "anchor": "Hello, greeting, hi there, good morning, nice to meet you",
        "label": "Greeting",
        "icon": "👋",
        "responses": [
            "こんにちは！Pentagonです。何でもお気軽にご質問ください。",
            "はじめまして！私はPentagonです。どのようなお手伝いができますか？",
        ]
    },
    "python_error": {
        "anchor": "Python error, bug, exception, traceback, crash, debugging code",
        "label": "Python Debug",
        "icon": "🐍",
        "responses": [
            "Pythonのエラーですね。エラーメッセージ（Traceback）を見せていただけますか？多くの場合、最後の行に原因が書かれています。",
            "コードのバグを見つけましょう。まずは `print()` で変数の中身を確認するか、`try-except` でエラーの種類を特定するのが第一歩です。",
        ]
    },
    "database": {
        "anchor": "Database connection error, SQL query, PostgreSQL, MySQL, DB credentials, port 5432",
        "label": "Database",
        "icon": "🗄️",
        "responses": [
            "データベース接続の問題ですね。まずは以下を確認してください：\n1. DBサーバーは起動していますか？\n2. ホスト名・ポート番号は正しいですか？\n3. 認証情報（ユーザー名・パスワード）に誤りはありませんか？",
            "DB関連のトラブルシューティングですね。`connection refused` が出ている場合、ポートが開いていないか、ファイアウォールでブロックされている可能性が高いです。",
        ]
    },
    "network": {
        "anchor": "Network issue, internet connection, DNS, timeout, firewall, proxy, VPN",
        "label": "Network",
        "icon": "🌐",
        "responses": [
            "ネットワークの問題のようですね。まずは `ping` や `traceroute` で接続先への到達性を確認しましょう。DNSの設定やプロキシの設定も見直してみてください。",
            "接続タイムアウトが発生していますか？VPNやファイアウォールの設定が原因であることが多いです。一時的にVPNをオフにして試してみてください。",
        ]
    },
    "git": {
        "anchor": "Git version control, merge conflict, branch, commit, push, pull request",
        "label": "Git / VCS",
        "icon": "🔀",
        "responses": [
            "Gitのトラブルですね。マージコンフリクトが起きている場合は、`git status` で競合ファイルを確認し、手動で修正した後 `git add` → `git commit` で解決できます。",
            "ブランチ管理でお困りですか？`git log --oneline --graph` でコミット履歴を可視化すると、状況が整理しやすくなります。",
        ]
    },
    "deployment": {
        "anchor": "Deploy, production server, Docker, Kubernetes, CI/CD, cloud hosting, AWS, GCP",
        "label": "Deploy / Infra",
        "icon": "🚀",
        "responses": [
            "デプロイに関するご質問ですね。Dockerを使用している場合、まずは `docker logs <container_id>` でコンテナのログを確認しましょう。ビルドの問題かランタイムの問題かで対処法が変わります。",
            "本番環境への反映ですね。CI/CDパイプラインが設定されていれば、`main` ブランチへのマージで自動デプロイが走るはずです。パイプラインのログを確認してみてください。",
        ]
    },
    "performance": {
        "anchor": "Performance optimization, slow, memory leak, CPU usage, latency, speed up",
        "label": "Performance",
        "icon": "⚡",
        "responses": [
            "パフォーマンスの問題ですね。まずはボトルネックの特定が重要です。Pythonなら `cProfile` や `line_profiler`、Webアプリなら開発者ツールの Performance タブで計測を始めましょう。",
            "メモリリークの可能性がありますか？`tracemalloc` モジュールでメモリ使用量の推移を追跡できます。長時間動作で徐々にメモリが増えていないか確認してみてください。",
        ]
    },
    "thanks": {
        "anchor": "Thank you, thanks, solved, fixed, resolved, it works now, great",
        "label": "Resolution",
        "icon": "✅",
        "responses": [
            "解決して良かったです！他にもお困りのことがあれば、いつでもお声がけください。",
            "お役に立てて嬉しいです！今後同じ問題が起きないよう、原因と解決策をメモしておくと良いですよ。",
        ]
    },
}

# ========== Pentagon エンジン ==========
class PentagonEngine:
    """Pentagon: 余熱(Thermal Trace)付き概念核チャットエンジン"""
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.dim = 384
        
        # アンカーの構築
        self.domains = list(KNOWLEDGE_DOMAINS.keys())
        anchor_texts = [KNOWLEDGE_DOMAINS[d]["anchor"] for d in self.domains]
        raw_emb = encoder.encode(anchor_texts)
        self.anchors = torch.tensor(raw_emb).float()
        self.anchors = self.anchors / self.anchors.norm(dim=1, keepdim=True)
        
        # 物理エンジン
        p = HDParams(
            dim=self.dim, use_ridge=False,
            sigma=0.3, sigma_rep=1.0, alpha_rep=0.5,
            marble_lr=0.5, marble_steps=100,
            marble_max_v=1.0, marble_momentum=0.2
        )
        self.core = HighDimConceptCore(p)
        self.core._centers = self.anchors.clone()
        self.core._weights = torch.ones(len(self.anchors)) * 0.5
        
        # 余熱システム
        self.thermal_traces = []
        self.heat_decay = 0.6
        self.heat_sigma = 3.0
        self.initial_heat = 30.0
        
        # 統計
        self.total_steps_history = []
    
    def process_input(self, user_text):
        """ユーザー入力を処理し、概念核で推論する"""
        # エンコード
        emb = self.encoder.encode([user_text])
        user_vec = torch.tensor(emb).float()
        user_vec = user_vec / user_vec.norm(dim=1, keepdim=True)
        user_vec = user_vec.squeeze(0)
        
        # 余熱の減衰
        for trace in self.thermal_traces:
            trace['heat'] *= self.heat_decay
        # 冷え切った余熱を除去
        self.thermal_traces = [t for t in self.thermal_traces if t['heat'] > 0.01]
        
        # 物理シミュレーション
        x = user_vec.clone().requires_grad_(True)
        v = torch.zeros_like(x)
        
        for step in range(self.core.p.marble_steps):
            # 基本ポテンシャル
            H_base = self.core.potential(x.unsqueeze(0)).squeeze()
            
            # 余熱ポテンシャル
            H_thermal = torch.tensor(0.0)
            for trace in self.thermal_traces:
                dist_sq = torch.sum((x - trace['pos'])**2)
                H_thermal = H_thermal - trace['heat'] * torch.exp(-dist_sq / (2 * self.heat_sigma**2))
            
            H_total = H_base + H_thermal
            H_total.backward()
            grad = x.grad.detach()
            x.grad.zero_()
            
            v = self.core.p.marble_momentum * v - self.core.p.marble_lr * grad
            v_norm = v.norm()
            if v_norm > self.core.p.marble_max_v:
                v = v * (self.core.p.marble_max_v / v_norm)
            
            x_new = x.detach() + v
            x_new = x_new / x_new.norm()
            x = x_new.requires_grad_(True)
        
        final_pos = x.detach()
        
        # 余熱を残す
        self.thermal_traces.append({'pos': final_pos.clone(), 'heat': self.initial_heat})
        
        # 最近傍の概念を特定
        dists = torch.norm(self.core._centers - final_pos.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(dists).item()
        domain_key = self.domains[nearest_idx]
        domain = KNOWLEDGE_DOMAINS[domain_key]
        
        # 全アンカーまでの距離（サイドバー用）
        all_dists = {}
        for i, d in enumerate(self.domains):
            all_dists[d] = dists[i].item()
        
        # 応答を選択
        import random
        response_text = random.choice(domain["responses"])
        
        return {
            "domain_key": domain_key,
            "domain": domain,
            "response": response_text,
            "distances": all_dists,
            "thermal_count": len(self.thermal_traces),
            "active_heats": [(t['heat'], self.domains[torch.argmin(torch.norm(self.core._centers - t['pos'].unsqueeze(0), dim=1)).item()]) for t in self.thermal_traces],
        }

# ========== Streamlit UI ==========
def main():
    st.set_page_config(
        page_title="Pentagon - Concept Core AI",
        page_icon="⬠",
        layout="wide",
    )
    
    # カスタムCSS
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.0em;
        margin-top: 0;
        margin-bottom: 2em;
    }
    .domain-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        margin: 2px;
    }
    .heat-bar {
        background: linear-gradient(90deg, #ff6b35, #ffd166, #06d6a0);
        height: 8px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ヘッダー
    st.markdown('<div class="main-title">⬠ Pentagon</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Physics-Based Concept Core AI — Powered by Gravity, Heat & Potential Fields</div>', unsafe_allow_html=True)
    
    # エンジンの初期化 (セッション状態で保持)
    if "engine" not in st.session_state:
        with st.spinner("Pentagon Engine starting up... (Loading SentenceTransformer)"):
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            st.session_state.engine = PentagonEngine(encoder)
            st.session_state.messages = []
            st.session_state.debug_info = None
    
    engine = st.session_state.engine
    
    # ========== サイドバー: デバッグパネル ==========
    with st.sidebar:
        st.markdown("### 🧠 Pentagon Internal State")
        st.markdown("---")
        
        # 余熱マップ
        st.markdown("#### 🌡️ Thermal Traces (余熱マップ)")
        if engine.thermal_traces:
            for i, trace in enumerate(engine.thermal_traces):
                nearest_idx = torch.argmin(torch.norm(engine.core._centers - trace['pos'].unsqueeze(0), dim=1)).item()
                domain_key = engine.domains[nearest_idx]
                domain = KNOWLEDGE_DOMAINS[domain_key]
                heat_pct = min(trace['heat'] / engine.initial_heat * 100, 100)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"{domain['icon']} {domain['label']}")
                    st.progress(heat_pct / 100)
                with col2:
                    st.caption(f"{trace['heat']:.1f}")
        else:
            st.caption("(No thermal traces yet)")
        
        st.markdown("---")
        
        # 最新の推論結果
        st.markdown("#### 📊 Last Inference")
        if st.session_state.debug_info:
            info = st.session_state.debug_info
            st.markdown(f"**Converged Valley:** {info['domain']['icon']} {info['domain']['label']}")
            
            st.markdown("**Distance to all concepts:**")
            sorted_dists = sorted(info['distances'].items(), key=lambda x: x[1])
            for d_key, dist in sorted_dists:
                d = KNOWLEDGE_DOMAINS[d_key]
                bar_len = max(0, int((1.5 - dist) / 1.5 * 100))
                st.caption(f"{d['icon']} {d['label']}: {dist:.3f}")
                st.progress(min(bar_len / 100, 1.0))
        else:
            st.caption("(No inference yet)")
        
        st.markdown("---")
        st.markdown("#### ℹ️ About Pentagon")
        st.caption(
            "Pentagon uses a physics-based 'Concept Core' engine. "
            "User input is projected into a 384-dimensional potential field. "
            "A marble rolls down the landscape, guided by gravity (attractors) "
            "and thermal traces (context memory), to find the most stable valley (concept)."
        )
    
    # ========== メインチャットエリア ==========
    # チャット履歴の表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="⬠" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])
    
    # ユーザー入力
    if prompt := st.chat_input("Pentagonに質問する..."):
        # ユーザーメッセージの表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Pentagon エンジンで推論
        result = engine.process_input(prompt)
        st.session_state.debug_info = result
        
        # 応答の構築
        domain = result["domain"]
        response = f"{domain['icon']} **[{domain['label']}]**\n\n{result['response']}"
        
        # アシスタントメッセージの表示
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="⬠"):
            st.markdown(response)
        
        # サイドバーの更新のためにリラン
        st.rerun()

if __name__ == "__main__":
    main()
