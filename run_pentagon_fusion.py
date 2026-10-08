import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --------------------------
# 1. パスと設定
# --------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TEXTS_PATH = os.path.join(OUTPUT_DIR, "wikipedia_texts.pkl")
# Approach Bで作った5000件のFAISSではなく、巨大地形構築フェーズで作った「100個の谷のFAISS」を使う
FAISS_LANDSCAPE_PATH = os.path.join(OUTPUT_DIR, "pentagon_landscape_massive.faiss")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 2. モデル定義 (Concept Decoder)
# --------------------------
class ConceptDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, concept_dim=384):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.concept_proj = nn.Linear(concept_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, concept_vec):
        embedded = self.embedding(x)
        h0 = self.concept_proj(concept_vec).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(embedded, (h0, c0))
        return self.fc(out)

def train_and_fuse():
    print("==================================================")
    print("Pentagon Fusion: Massive Landscape + Direct Projection")
    print("==================================================")
    
    # --------------------------
    # 3. データと辞書の準備 (Decoder用)
    # --------------------------
    print("Loading text data for Decoder training...")
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)[:2000] # 高速化のため2000件
        
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    concept_vectors = encoder.encode(texts, show_progress_bar=False)
    
    chars = set("".join(texts))
    for special in ["<PAD>", "<SOS>", "<EOS>"]: chars.add(special)
    char2idx = {c: i for i, c in enumerate(list(chars))}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(char2idx)
    
    # --------------------------
    # 4. Decoderの高速学習
    # --------------------------
    model = ConceptDecoder(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    max_len = 50
    X_data, Y_data = [], []
    for t in texts:
        seq = [char2idx.get(c, char2idx["<PAD>"]) for c in t[:max_len]]
        x_seq = [char2idx["<SOS>"]] + seq
        y_seq = seq + [char2idx["<EOS>"]]
        while len(x_seq) < max_len + 1:
            x_seq.append(char2idx["<PAD>"])
            y_seq.append(char2idx["<PAD>"])
        X_data.append(x_seq)
        Y_data.append(y_seq)
        
    X_tensor = torch.tensor(X_data, dtype=torch.long).to(device)
    Y_tensor = torch.tensor(Y_data, dtype=torch.long).to(device)
    C_tensor = torch.tensor(concept_vectors, dtype=torch.float32).to(device)
    
    epochs = 40
    batch_size = 64
    print(f"\nTraining Neural Decoder on {len(texts)} texts...")
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X_tensor), batch_size):
            x_batch = X_tensor[i:i+batch_size]
            y_batch = Y_tensor[i:i+batch_size]
            c_batch = C_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = model(x_batch, c_batch)
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_tensor):.4f}")
            
    print(f"Decoder trained in {time.time() - start_time:.1f} seconds.")
    
    # --------------------------
    # 5. 融合テスト (巨大地形 -> 谷の特定 -> 直接投影)
    # --------------------------
    print("\nLoading Massive Landscape (100 Valleys) FAISS Index...")
    if not os.path.exists(FAISS_LANDSCAPE_PATH):
        print("[ERROR] Landscape FAISS index not found.")
        return
        
    landscape_index = faiss.read_index(FAISS_LANDSCAPE_PATH)
    
    def decode_from_vector(concept_vec, max_len=40):
        model.eval()
        c_tensor = torch.tensor(concept_vec, dtype=torch.float32).unsqueeze(0).to(device)
        current_input = torch.tensor([[char2idx["<SOS>"]]], dtype=torch.long).to(device)
        
        h0 = model.concept_proj(c_tensor).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        generated_text = ""
        for _ in range(max_len):
            embedded = model.embedding(current_input)
            out, (h0, c0) = model.lstm(embedded, (h0, c0))
            logits = model.fc(out)
            next_char_idx = logits.argmax(dim=-1).item()
            next_char = idx2char[next_char_idx]
            if next_char == "<EOS>": break
            generated_text += next_char
            current_input = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
        return generated_text

    print("\n==================================================")
    print("FUSION TEST: Input -> Gravity to Valley -> Neural Decoding")
    print("==================================================")
    
    test_queries = ["宇宙の起源について", "日本の文化と歴史", "最新のコンピュータ技術"]
    
    for q in test_queries:
        # 1. ユーザー入力をベクトル化
        q_vec = encoder.encode([q]).astype(np.float32)
        faiss.normalize_L2(q_vec)
        
        # 2. 地形の重力計算: 最も近い「巨大な谷（アンカー）」に転がり落ちる
        distances, indices = landscape_index.search(q_vec, 1)
        valley_id = indices[0][0]
        
        # 谷の底（アンカーベクトル）を取得
        valley_vector = landscape_index.reconstruct(int(valley_id))
        
        # 3. 谷の座標を直接Decoderに投影し、文字を紡ぐ
        decoded_text = decode_from_vector(valley_vector)
        
        print(f"\n[ユーザー入力] : {q}")
        print(f"[重力降下]     : 巨大地形の 谷 #{valley_id} へ到達 (距離: {distances[0][0]:.4f})")
        print(f"[直接脳内投影] : {decoded_text}")

if __name__ == "__main__":
    train_and_fuse()
