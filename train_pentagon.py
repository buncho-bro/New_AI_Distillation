import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pentagon_landscape.pt")

def load_training_data():
    """
    本格的な学習用データセットをロード。
    本番環境ではWikipediaダンプや社内ドキュメントを読み込む。
    ここでは実証のため、多数の合成・汎用テキストを使用。
    """
    print("Loading training data...")
    import random
    topics = {
        'Technology & AI': (['Apple', 'Microsoft', 'Google', 'AI', 'Software', 'The developer'], ['released', 'announced', 'updated', 'developed', 'compiled'], ['a new smartphone', 'an AI model', 'a cloud service', 'an operating system', 'the source code']),
        'Space Exploration': (['NASA', 'SpaceX', 'Astronauts', 'The rover', 'A telescope'], ['launched', 'discovered', 'landed on', 'photographed', 'explored'], ['a new planet', 'Mars', 'the moon', 'a black hole', 'the space station']),
        'Politics & Economy': (['The president', 'The senator', 'The government', 'The committee', 'The central bank'], ['signed', 'vetoed', 'discussed', 'proposed', 'raised'], ['a new law', 'the budget', 'a trade agreement', 'the healthcare bill', 'interest rates']),
        'Sports & Olympics': (['The team', 'The striker', 'The athlete', 'The champion', 'The swimmer'], ['won', 'lost', 'scored', 'defended', 'broke'], ['the gold medal', 'the world cup', 'the final match', 'the championship title', 'the world record']),
        'Cooking & Food': (['The chef', 'The baker', 'My mother', 'The restaurant'], ['cooked', 'baked', 'prepared', 'served'], ['a delicious pizza', 'a chocolate cake', 'a healthy salad', 'spicy curry']),
        'Music & Arts': (['The band', 'The singer', 'The artist', 'The orchestra'], ['performed', 'released', 'composed', 'painted'], ['a new album', 'a beautiful song', 'a masterpiece', 'a symphony'])
    }
    
    texts = []
    # 各トピック1000件、合計6000件のデータを生成
    for topic_name, parts in topics.items():
        s, v, o = parts
        for _ in range(1000):
            text = f"{random.choice(s)} {random.choice(v)} {random.choice(o)}."
            # 多言語空間をより豊かにするため、英語と日本語のノイズ（タイポ）を混合
            noise = [" Yesterday.", " In a surprising turn of events.", " Experts say this is huge.", 
                     " 全く驚きですね。", " 昨日のニュースです。", " 専門家も注目しています。", " 楽しみに待っています。"]
            text += random.choice(noise)
            texts.append((text, topic_name))
            
    random.shuffle(texts)
    return [t[0] for t in texts], [t[1] for t in texts]

def train_landscape(n_valleys=150, dim=384):
    """
    大量のデータから自律的に谷（アンカー）を形成し、地形ファイルとして保存する。
    """
    raw_texts, original_topics = load_training_data()
    print(f"Loaded {len(raw_texts)} documents for training.")
    
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Encoding documents into vector space...")
    embeddings = encoder.encode(raw_texts, show_progress_bar=True, batch_size=256)
    
    print(f"Running clustering to discover {n_valleys} natural concepts (valleys)...")
    clustering = MiniBatchKMeans(n_clusters=n_valleys, random_state=42, batch_size=1024)
    clustering.fit(embeddings)
    
    centroids = clustering.cluster_centers_
    anchors = torch.tensor(centroids).float()
    anchors = anchors / anchors.norm(dim=1, keepdim=True)
    
    # 谷の深さ（重み）の計算：多くのドキュメントが集まった谷ほど深くなる
    _, counts = np.unique(clustering.labels_, return_counts=True)
    weights = torch.tensor(counts).float()
    weights = (weights / weights.max()) * 2.0  # 正規化 (Max 2.0)
    
    # AIが発見した各谷に、一番多く含まれている本来のトピックを使って仮の名前をつける
    domain_names = []
    for i in range(n_valleys):
        cluster_indices = np.where(clustering.labels_ == i)[0]
        if len(cluster_indices) > 0:
            topics_in_cluster = [original_topics[idx] for idx in cluster_indices]
            # 最頻値を取得
            most_common = max(set(topics_in_cluster), key=topics_in_cluster.count)
            domain_names.append(f"[{most_common}] Sub-concept #{i}")
        else:
            domain_names.append(f"Unknown Concept #{i}")
            
    print("Saving landscape...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = {
        'anchors': anchors,
        'weights': weights,
        'domain_names': domain_names
    }
    torch.save(data, OUTPUT_PATH)
    print(f"\n[SUCCESS] Successfully built and saved Pentagon Landscape to {OUTPUT_PATH} !")
    print("You can now load this brain into PentagonEngine for production inference.")

if __name__ == "__main__":
    train_landscape(n_valleys=150)
