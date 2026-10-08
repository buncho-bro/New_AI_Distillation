import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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

class PentagonFusionCore:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.faiss_path = os.path.join(models_dir, "pentagon_landscape_massive.faiss")
        self.model_path = os.path.join(models_dir, "concept_decoder.pth")
        self.dict_path = os.path.join(models_dir, "decoder_char2idx.pkl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder = None
        self.landscape = None
        self.decoder = None
        self.char2idx = None
        self.idx2char = None

    def load(self):
        print("Loading dependencies...")
        self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.landscape = faiss.read_index(self.faiss_path)
        
        with open(self.dict_path, "rb") as f:
            self.char2idx = pickle.load(f)
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        
        vocab_size = len(self.char2idx)
        self.decoder = ConceptDecoder(vocab_size).to(self.device)
        self.decoder.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.decoder.eval()
        print("Core loaded successfully.")

    def query(self, user_text, max_len=60):
        # 1. ユーザー入力をベクトル化
        q_vec = self.encoder.encode([user_text]).astype(np.float32)
        faiss.normalize_L2(q_vec)
        
        # 2. 地形の重力計算
        distances, indices = self.landscape.search(q_vec, 1)
        valley_id = indices[0][0]
        valley_vector = self.landscape.reconstruct(int(valley_id))
        
        # 3. 直接脳内投影 (Direct Projection Decoding)
        c_tensor = torch.tensor(valley_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        current_input = torch.tensor([[self.char2idx["<SOS>"]]], dtype=torch.long).to(self.device)
        
        h0 = self.decoder.concept_proj(c_tensor).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        generated_text = ""
        for _ in range(max_len):
            embedded = self.decoder.embedding(current_input)
            out, (h0, c0) = self.decoder.lstm(embedded, (h0, c0))
            logits = self.decoder.fc(out)
            next_char_idx = logits.argmax(dim=-1).item()
            next_char = self.idx2char[next_char_idx]
            
            if next_char == "<EOS>": break
            generated_text += next_char
            current_input = torch.tensor([[next_char_idx]], dtype=torch.long).to(self.device)
            
        return generated_text, valley_id, distances[0][0]
