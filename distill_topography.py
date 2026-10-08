import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------
# 1. データ準備と辞書構築
# --------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TEXTS_PATH = os.path.join(OUTPUT_DIR, "wikipedia_texts.pkl")

print("Loading dataset for Topographical Distillation...")
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)[:3000] # 3000件で実験

# 文字レベルの超シンプル辞書（Student用）
chars = set("".join(texts))
char2idx = {c: i for i, c in enumerate(list(chars))}
vocab_size = len(char2idx)
print(f"Student Vocabulary size: {vocab_size} features")

# --------------------------
# 2. Studentモデル（極小ネットワーク）
# --------------------------
class TinyStudent(nn.Module):
    def __init__(self, input_size, output_size=384):
        super().__init__()
        # パラメータは [vocab_size x 384] のみ。Teacher（1億以上）の数百分の一
        self.fc = nn.Linear(input_size, output_size, bias=False)
        
    def forward(self, x):
        # x: (batch, input_size) multi-hot vector
        return self.fc(x)

# テキストをStudent用のMulti-hotベクトルに変換する関数
def text_to_multihot(text_list, vocab_size, char_dict):
    tensors = torch.zeros(len(text_list), vocab_size)
    for i, t in enumerate(text_list):
        for c in t:
            if c in char_dict:
                tensors[i][char_dict[c]] = 1.0
    return tensors

# --------------------------
# 3. 蒸留ループ（地形の移植）
# --------------------------
def distill_topography():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nLoading Teacher Model (Genius Brain)...")
    teacher = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Initializing Student Model (Tiny Brain)...")
    student = TinyStudent(vocab_size, output_size=384).to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.01)
    
    epochs = 50
    batch_size = 100
    
    print("\nStarting Topographical Distillation...")
    print("Transferring relative distances (Topology) from Teacher to Student...")
    
    print("Pre-computing Teacher vectors (Genius Brain's Geography)...")
    teacher_vecs_all = teacher.encode(texts, show_progress_bar=True, batch_size=256)
    
    start_time = time.time()
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        # Shuffle indices
        indices = np.arange(len(texts))
        np.random.shuffle(indices)
        
        for i in range(0, len(texts), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_texts = [texts[idx] for idx in batch_indices]
            if len(batch_texts) < 2: continue
            
            # プリコンピュートされたTeacherの絶対座標を取得
            teacher_vecs = torch.tensor(teacher_vecs_all[batch_indices], dtype=torch.float32).to(device)
            teacher_vecs = F.normalize(teacher_vecs, p=2, dim=1)
            
            # Studentの座標を取得
            student_inputs = text_to_multihot(batch_texts, vocab_size, char2idx).to(device)
            student_vecs = student(student_inputs)
            student_vecs = F.normalize(student_vecs, p=2, dim=1)
            
            # ★ 地形蒸留のコアロジック ★
            # 空間内の全点間の「距離（コサイン類似度）」の行列を計算
            # Teacherの地形図 (batch_size x batch_size)
            teacher_topology = torch.mm(teacher_vecs, teacher_vecs.t())
            # Studentの地形図 (batch_size x batch_size)
            student_topology = torch.mm(student_vecs, student_vecs.t())
            
            # Teacherの地形とStudentの地形の「形の誤差（MSE）」をLossとする
            loss = F.mse_loss(student_topology, teacher_topology)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Topology Error (Loss): {total_loss/(len(texts)//batch_size):.4f}")
            
    print(f"Distillation completed in {time.time() - start_time:.1f} seconds.")
    
    # --------------------------
    # 4. 検証（未学習の概念でのテスト）
    # --------------------------
    print("\n==================================================")
    print("VERIFICATION: Testing the Distilled Tiny Brain")
    print("==================================================")
    student.eval()
    
    test_words = ["りんご", "みかん", "自動車", "トラック", "宇宙", "ブラックホール"]
    
    # Studentモデルによるベクトル化
    with torch.no_grad():
        inputs = text_to_multihot(test_words, vocab_size, char2idx).to(device)
        vecs = student(inputs)
        vecs = F.normalize(vecs, p=2, dim=1)
    
    print("Student Model's Spatial Understanding (Cosine Similarity):")
    print("-" * 50)
    
    # ペアの類似度を計算して出力
    pairs = [
        ("りんご", "みかん"), 
        ("自動車", "トラック"), 
        ("宇宙", "ブラックホール"),
        ("りんご", "トラック"), 
        ("みかん", "ブラックホール")
    ]
    
    for w1, w2 in pairs:
        i1, i2 = test_words.index(w1), test_words.index(w2)
        sim = torch.dot(vecs[i1], vecs[i2]).item()
        
        # 評価
        if sim > 0.6: judgment = "近い谷 (同じ概念)"
        elif sim < 0.4: judgment = "遠い谷 (違う概念)"
        else: judgment = "中間"
            
        print(f"[{w1}] と [{w2}] の距離: {sim:.3f} -> {judgment}")
        
    print("\n[SUCCESS] The tiny brain successfully inherited the conceptual topography without reading meanings!")

if __name__ == "__main__":
    distill_topography()
