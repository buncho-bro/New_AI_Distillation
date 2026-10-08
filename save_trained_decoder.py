import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TEXTS_PATH = os.path.join(OUTPUT_DIR, "wikipedia_texts.pkl")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "concept_decoder.pth")
DICT_SAVE_PATH = os.path.join(OUTPUT_DIR, "decoder_char2idx.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_and_save():
    print("Loading text data for persistent Decoder training...")
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)[:2000]
        
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    concept_vectors = encoder.encode(texts, show_progress_bar=False)
    
    chars = set("".join(texts))
    for special in ["<PAD>", "<SOS>", "<EOS>"]: chars.add(special)
    char2idx = {c: i for i, c in enumerate(list(chars))}
    vocab_size = len(char2idx)
    
    # 辞書の保存
    with open(DICT_SAVE_PATH, "wb") as f:
        pickle.dump(char2idx, f)
    
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
    
    epochs = 50
    batch_size = 64
    print(f"Training Neural Decoder on {len(texts)} texts...")
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
    
    # モデルの保存
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[SUCCESS] Saved persistent brain to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_save()
