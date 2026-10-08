import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------
# 1. データ準備と辞書構築
# --------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TEXTS_PATH = os.path.join(OUTPUT_DIR, "wikipedia_texts.pkl")

print("Loading dataset...")
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)[:1000] # 学習を早くするため1000件に絞る

print("Encoding concept vectors (Teacher inputs)...")
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
concept_vectors = encoder.encode(texts, show_progress_bar=False) # (1000, 384)

# 超シンプルな文字レベル(Char-level)辞書の構築
chars = set("".join(texts))
chars.add("<PAD>")
chars.add("<SOS>") # Start of sequence
chars.add("<EOS>") # End of sequence
char2idx = {c: i for i, c in enumerate(list(chars))}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx)
print(f"Vocabulary size (Unique characters): {vocab_size}")

# --------------------------
# 2. PyTorch モデル定義 (Approach A)
# --------------------------
class ConceptDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, concept_dim=384):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=char2idx["<PAD>"])
        # Projection Layer: 384次元の概念空間 -> RNNの思考空間(256次元)へ直接変換
        self.concept_proj = nn.Linear(concept_dim, hidden_dim)
        
        # 文章を紡ぐRNN
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, concept_vec):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        
        # ★アプローチAの真髄★
        # 概念ベクトル(384d)をLSTMの「初期の思考(h0)」として直接脳に叩き込む
        h0 = self.concept_proj(concept_vec).unsqueeze(0) # (1, batch, hidden_dim)
        c0 = torch.zeros_like(h0)
        
        out, _ = self.lstm(embedded, (h0, c0))
        logits = self.fc(out) # (batch, seq_len, vocab_size)
        return logits

# --------------------------
# 3. 学習ループ
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConceptDecoder(vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 学習データのテンソル化
max_len = 50 # 最初の50文字だけ学習
X_data, Y_data = [], []
for t in texts:
    # 入力: <SOS> + text
    # 正解: text + <EOS>
    seq = [char2idx.get(c, char2idx["<PAD>"]) for c in t[:max_len]]
    x_seq = [char2idx["<SOS>"]] + seq
    y_seq = seq + [char2idx["<EOS>"]]
    
    # Padding
    while len(x_seq) < max_len + 1:
        x_seq.append(char2idx["<PAD>"])
        y_seq.append(char2idx["<PAD>"])
        
    X_data.append(x_seq)
    Y_data.append(y_seq)

X_tensor = torch.tensor(X_data, dtype=torch.long).to(device)
Y_tensor = torch.tensor(Y_data, dtype=torch.long).to(device)
C_tensor = torch.tensor(concept_vectors, dtype=torch.float32).to(device)

epochs = 50
batch_size = 64
print("\nStarting Training (Projection Layer)...")
model.train()
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

# --------------------------
# 4. 推論（ベクトルから直接デコード）
# --------------------------
print("\n==================================================")
print("Testing Direct Vector-to-Text Decoding (Approach A)")
print("==================================================")
model.eval()

def decode_from_vector(concept_vec, max_len=30):
    c_tensor = torch.tensor(concept_vec, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 最初の入力は <SOS> のみ
    current_input = torch.tensor([[char2idx["<SOS>"]]], dtype=torch.long).to(device)
    
    # 概念ベクトルを初期状態にセット
    h0 = model.concept_proj(c_tensor).unsqueeze(0)
    c0 = torch.zeros_like(h0)
    
    generated_text = ""
    for _ in range(max_len):
        embedded = model.embedding(current_input)
        out, (h0, c0) = model.lstm(embedded, (h0, c0))
        
        logits = model.fc(out)
        next_char_idx = logits.argmax(dim=-1).item()
        next_char = idx2char[next_char_idx]
        
        if next_char == "<EOS>":
            break
            
        generated_text += next_char
        current_input = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
        
    return generated_text

test_queries = ["日本の歴史について", "コンピュータの仕組み", "美味しい料理の作り方"]
for q in test_queries:
    q_vec = encoder.encode([q])[0]
    result = decode_from_vector(q_vec)
    print(f"\n[Query]  : {q}")
    print(f"[Vector] : [{q_vec[0]:.3f}, {q_vec[1]:.3f}, ... {q_vec[-1]:.3f}] (384-dimensional Concept)")
    print(f"[Decode] : {result}")

print("\n[FINISH] Directly projected vectors into characters.")
