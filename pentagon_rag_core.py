import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 翻訳用に google-generativeai を試みる
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class PentagonRAGCore:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.faiss_path = os.path.join(models_dir, "wikipedia_full.faiss")
        self.texts_path = os.path.join(models_dir, "wikipedia_texts.pkl")
        
        self.encoder = None
        self.index = None
        self.texts = None
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if HAS_GENAI and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def load(self):
        if not os.path.exists(self.faiss_path) or not os.path.exists(self.texts_path):
            raise FileNotFoundError("FAISS index or texts pickle not found. Please run approach B test script first.")
            
        print("Loading SentenceTransformer...")
        self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("Loading FAISS Index...")
        self.index = faiss.read_index(self.faiss_path)
        print("Loading Texts...")
        with open(self.texts_path, "rb") as f:
            self.texts = pickle.load(f)
        print("Pentagon RAG Core loaded successfully.")

    def query(self, user_text, top_k=5):
        # 1. クエリのベクトル化
        vec = self.encoder.encode([user_text]).astype(np.float32)
        faiss.normalize_L2(vec)
        
        # 2. 逆写像 (Reverse-Mapping)
        distances, indices = self.index.search(vec, top_k)
        
        fragments = []
        for i in range(top_k):
            idx = indices[0][i]
            fragments.append(self.texts[idx])
            
        # 3. 翻訳フェーズ (LLM)
        if self.model:
            prompt = (
                "あなたは『概念の破片』から言語を紡ぐ翻訳機です。自身の事前学習知識は極力使わず、"
                "以下の【概念の破片】の情報を繋ぎ合わせて、ユーザーの意図に沿った自然な日本語の回答を生成してください。\n\n"
                f"【ユーザーの意図】\n{user_text}\n\n"
                "【概念の破片】\n"
            )
            for f in fragments:
                prompt += f"- {f}\n"
                
            try:
                response = self.model.generate_content(prompt)
                translated_text = response.text
                return translated_text, fragments
            except Exception as e:
                return f"[翻訳エラー] API呼び出しに失敗しました: {e}\n\n[抽出された生データ]\n" + "\n".join([f"- {f[:50]}..." for f in fragments]), fragments
        else:
            # APIキーがない場合のフォールバック（生の破片をそのまま返す）
            fallback = "※翻訳用LLMのAPIキー(GEMINI_API_KEY)が未設定のため、生の概念破片を出力します。\n\n"
            for i, f in enumerate(fragments):
                fallback += f"【破片 {i+1}】 {f}\n"
            return fallback, fragments
