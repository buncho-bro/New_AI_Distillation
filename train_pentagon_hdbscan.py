import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pentagon_landscape_hdbscan.pt")

def load_training_data():
    """
    KMeans版と同じ6000件のデータを生成
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
    for topic_name, parts in topics.items():
        s, v, o = parts
        for _ in range(1000):
            text = f"{random.choice(s)} {random.choice(v)} {random.choice(o)}."
            noise = [" Yesterday.", " In a surprising turn of events.", " Experts say this is huge.", 
                     " 全く驚きですね。", " 昨日のニュースです。", " 専門家も注目しています。", " 楽しみに待っています。"]
            text += random.choice(noise)
            texts.append((text, topic_name))
            
    random.shuffle(texts)
    return [t[0] for t in texts], [t[1] for t in texts]

def train_landscape_hdbscan(min_cluster_size=15):
    """
    HDBSCANを用いて、谷の数を指定せずに自律的に概念を発見する
    """
    raw_texts, original_topics = load_training_data()
    print(f"Loaded {len(raw_texts)} documents for training.")
    
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Encoding documents into vector space...")
    embeddings = encoder.encode(raw_texts, show_progress_bar=True, batch_size=256)
    
    print(f"Running HDBSCAN clustering (min_cluster_size={min_cluster_size}) to AUTONOMOUSLY discover valleys...")
    # HDBSCANによるクラスタリング
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1) # -1 はどの谷にも属さないノイズデータ
        
    n_valleys = len(unique_labels)
    print(f"[HDBSCAN Result] AI autonomously discovered {n_valleys} distinct valleys!")
    print(f"                 (Ignored {(labels == -1).sum()} noisy data points)")
    
    if n_valleys == 0:
        print("Error: No valleys found. Try decreasing min_cluster_size.")
        return

    # 各クラスタ（谷）の重心（アンカー）と重み（データ数）を計算
    anchors_list = []
    weights_list = []
    domain_names = []
    
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        # 重心ベクトル
        cluster_points = embeddings[indices]
        centroid = np.mean(cluster_points, axis=0)
        anchors_list.append(centroid)
        
        # 重み（データ数に基づく谷の深さ）
        weights_list.append(len(indices))
        
        # 谷の名前の決定（最頻値）
        topics_in_cluster = [original_topics[idx] for idx in indices]
        most_common = max(set(topics_in_cluster), key=topics_in_cluster.count)
        domain_names.append(f"[{most_common}] Auto-Concept #{label}")

    anchors = torch.tensor(np.array(anchors_list)).float()
    anchors = anchors / anchors.norm(dim=1, keepdim=True)
    
    weights = torch.tensor(weights_list).float()
    weights = (weights / weights.max()) * 2.0  # 正規化 (Max 2.0)
    
    print("Saving HDBSCAN landscape...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = {
        'anchors': anchors,
        'weights': weights,
        'domain_names': domain_names
    }
    torch.save(data, OUTPUT_PATH)
    print(f"\n[SUCCESS] Successfully built and saved HDBSCAN Landscape to {OUTPUT_PATH} !")

if __name__ == "__main__":
    train_landscape_hdbscan(min_cluster_size=20)
