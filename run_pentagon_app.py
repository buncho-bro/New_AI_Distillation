import os
import time
import sys
from pentagon_core import PentagonEngine

def get_base_path():
    """exe化(frozen)された場合と、スクリプトとして実行された場合の両方に対応するパス取得"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(get_base_path(), "models", "pentagon_landscape.pt")

def run_app():
    print("==========================================")
    print("   Pentagon Core - Production Terminal")
    print("==========================================\n")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Landscape file not found at {MODEL_PATH}")
        print("Please run 'python train_pentagon.py' first to build the brain.")
        return
        
    engine = PentagonEngine()
    engine.load_landscape(MODEL_PATH)
    
    print("\nSystem ready. Type your query, or 'exit' to quit.")
    print("If Pentagon makes a mistake, type 'wrong' immediately after to correct it and update the landscape.\n")
    
    last_query = None
    last_inferred_idx = None
    
    while True:
        try:
            user_input = input("\n[User]: ").strip()
        except EOFError:
            break
            
        if user_input.lower() in ['exit', 'quit']:
            # 終了時に成長した脳を保存
            engine.save_landscape(MODEL_PATH)
            print("Goodbye.")
            break
            
        if user_input.lower() == 'wrong':
            if last_query is None:
                print("[System] No previous query to correct.")
                continue
                
            print(f"[System] Initiating Reinforcement Learning for: '{last_query}'")
            print("Please tell me which concept it SHOULD have been mapped to.")
            # 簡易的に、最も関連しそうな谷をいくつか提示するか、テキストで探す
            # 今回は実装をシンプルにするため、ユーザーに正しいテキストを再入力させ、
            # 最も近い谷を見つけてそこを強化する。
            correct_text = input("[System] Type a clear sentence describing the correct topic: ").strip()
            # 一旦、その正しいテキストで推論(Thermal Traceなし)して目標の谷を探す
            target_idx, target_name = engine.inference(correct_text, use_thermal=False)
            
            print(f"[System] Ah, you meant it belongs to '{target_name}'. Updating landscape...")
            # フィードバック学習の実行（地形の変形）
            engine.learn_from_feedback(last_query, target_idx, shift_ratio=0.15, depth_boost=0.5)
            # 学習状態を即座に保存
            engine.save_landscape(MODEL_PATH)
            print("[System] Memory updated and saved.")
            continue
            
        # 通常の推論
        if not user_input:
            continue
            
        start_t = time.time()
        idx, domain = engine.inference(user_input, use_thermal=True)
        inf_time = time.time() - start_t
        
        print(f"  => [Pentagon] Mapped to: {domain}")
        print(f"     (Inference took {inf_time:.4f} sec)")
        
        last_query = user_input
        last_inferred_idx = idx

if __name__ == "__main__":
    run_app()
