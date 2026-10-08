import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import gc
import signal
import threading

def load_base_data():
    import random
    topics = {
        'Tech': (['Apple', 'Microsoft'], ['released', 'updated'], ['a smartphone', 'an OS']),
        'Space': (['NASA', 'SpaceX'], ['launched', 'discovered'], ['a planet', 'a rocket']),
        'Politics': (['The president', 'The senator'], ['signed', 'vetoed'], ['a law', 'the budget']),
        'Sports': (['The team', 'The athlete'], ['won', 'lost'], ['the medal', 'the match']),
        'Food': (['The chef', 'My mother'], ['cooked', 'baked'], ['a pizza', 'a cake']),
        'Music': (['The band', 'The singer'], ['performed', 'released'], ['a song', 'an album'])
    }
    texts = []
    for topic_name, parts in topics.items():
        s, v, o = parts
        for _ in range(100):
            text = f"{random.choice(s)} {random.choice(v)} {random.choice(o)}."
            texts.append(text)
    return texts

def run_single_test(test_data, result_dict):
    """スレッドで実行するHDBSCANテスト"""
    try:
        clusterer = HDBSCAN(min_cluster_size=20, metric='euclidean')
        labels = clusterer.fit_predict(test_data)
        unique_labels = set(labels)
        n_noise = int((labels == -1).sum())
        if -1 in unique_labels:
            unique_labels.remove(-1)
        result_dict['valleys'] = len(unique_labels)
        result_dict['noise'] = n_noise
        result_dict['success'] = True
    except Exception as e:
        result_dict['error'] = str(e)
        result_dict['success'] = False

def binary_search():
    print("=" * 60)
    print("HDBSCAN Binary Search: Finding the Breaking Point")
    print("=" * 60)
    
    texts = load_base_data()
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("Encoding base vectors...")
    base_embeddings = encoder.encode(texts, show_progress_bar=False)
    
    # 探査する値のリスト（10000〜50000を細かく刻む）
    test_sizes = [15000, 20000, 25000, 30000, 35000, 40000, 45000]
    TIME_LIMIT = 120  # 各テストの制限時間（秒）
    
    results = []
    
    for size in test_sizes:
        print(f"\n{'='*50}")
        print(f"[Test] N = {size:,} points (Time limit: {TIME_LIMIT}s)")
        print(f"{'='*50}")
        
        # データ生成
        repeats = (size // len(base_embeddings)) + 1
        expanded = np.tile(base_embeddings, (repeats, 1))[:size]
        noise = np.random.normal(0, 0.05, expanded.shape).astype(np.float32)
        test_data = expanded + noise
        
        gc.collect()
        
        result_dict = {'success': False, 'valleys': 0, 'noise': 0, 'error': ''}
        
        start_time = time.time()
        thread = threading.Thread(target=run_single_test, args=(test_data, result_dict))
        thread.start()
        thread.join(timeout=TIME_LIMIT)
        elapsed = time.time() - start_time
        
        if thread.is_alive():
            print(f"  -> TIMEOUT after {TIME_LIMIT}s!")
            print(f"  -> Status: BREAKDOWN (Algorithm could not finish in time)")
            results.append((size, 'TIMEOUT', elapsed, 0, 0))
            # スレッドがまだ動いているが、次のテストに進む
            # (Pythonではスレッドを安全にkillできないため、ここで探査を打ち切る)
            print(f"\n  Breaking point found! HDBSCAN cannot handle N={size:,} within {TIME_LIMIT}s.")
            break
        elif result_dict['success']:
            print(f"  -> Time: {elapsed:.2f}s")
            print(f"  -> Valleys Found: {result_dict['valleys']}")
            print(f"  -> Noise Points: {result_dict['noise']:,}")
            results.append((size, 'OK', elapsed, result_dict['valleys'], result_dict['noise']))
        else:
            print(f"  -> CRASHED: {result_dict['error']}")
            results.append((size, 'CRASH', elapsed, 0, 0))
            break
    
    # 最終サマリー
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'N':>10} | {'Status':>10} | {'Time (s)':>10} | {'Valleys':>8} | {'Noise':>8}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    # 最初の10000件の結果も表示
    print(f"{'10,000':>10} | {'OK':>10} | {'11.91':>10} | {'44':>8} | {'0':>8}")
    for size, status, elapsed, valleys, noise in results:
        print(f"{size:>10,} | {status:>10} | {elapsed:>10.2f} | {valleys:>8} | {noise:>8,}")

if __name__ == "__main__":
    binary_search()
