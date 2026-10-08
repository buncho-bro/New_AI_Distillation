import os
import sys
import time
import threading
import customtkinter as ctk
from pentagon_core import PentagonEngine

# --- パス解決 ---
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(get_base_path(), "models", "pentagon_landscape.pt")

# --- UI設定 ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PentagonGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pentagon Core - Concept Inference")
        self.geometry("700x800")
        
        # エンジン関連
        self.engine = None
        self.last_query = None
        self.is_learning_mode = False
        
        self.setup_ui()
        
        # 非同期でエンジンをロード
        self.append_log("System", "Booting Pentagon Core... Please wait.", color="#50b3a2")
        threading.Thread(target=self.init_engine, daemon=True).start()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # チャットログ表示エリア
        self.textbox = ctk.CTkTextbox(self, state="disabled", font=("Meiryo", 14), wrap="word")
        self.textbox.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        
        # 入力エリアのフレーム
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your text here...", font=("Meiryo", 14), height=40)
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", self.on_send)
        
        self.send_button = ctk.CTkButton(self.input_frame, text="Send", font=("Meiryo", 14, "bold"), width=80, height=40, command=self.on_send)
        self.send_button.grid(row=0, column=1)

    def append_log(self, sender, message, color=None):
        self.textbox.configure(state="normal")
        # タグを使用して色を付ける設定
        tag_name = f"tag_{sender}"
        self.textbox.tag_config(tag_name, foreground=color if color else "#FFFFFF")
        
        self.textbox.insert("end", f"[{sender}]\n", tag_name)
        self.textbox.insert("end", f"{message}\n\n")
        self.textbox.yview("end")
        self.textbox.configure(state="disabled")

    def init_engine(self):
        try:
            self.engine = PentagonEngine()
            if os.path.exists(MODEL_PATH):
                self.engine.load_landscape(MODEL_PATH)
                self.append_log("System", "Brain loaded successfully. System ready.", color="#50b3a2")
            else:
                self.append_log("System", f"Error: Landscape not found at {MODEL_PATH}.", color="#ff5555")
        except Exception as e:
            self.append_log("System", f"Initialization Error: {e}", color="#ff5555")

    def on_send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
            
        self.entry.delete(0, "end")
        self.append_log("User", text, color="#66b3ff")
        
        if self.engine is None:
            self.append_log("System", "Engine is still loading. Please wait.", color="#ff5555")
            return
            
        # UIを固めないように別スレッドで処理
        if self.is_learning_mode:
            threading.Thread(target=self.process_learning, args=(text,), daemon=True).start()
        elif text.lower() == 'wrong':
            if self.last_query is None:
                self.append_log("System", "No previous query to correct.", color="#ffaa00")
            else:
                self.is_learning_mode = True
                self.append_log("Pentagon", f"I'm sorry. Initiating Reinforcement Learning for: '{self.last_query}'.\nPlease describe the CORRECT topic/concept.", color="#ffaa00")
        else:
            threading.Thread(target=self.process_inference, args=(text,), daemon=True).start()

    def process_inference(self, text):
        start_t = time.time()
        idx, domain = self.engine.inference(text, use_thermal=True)
        inf_time = time.time() - start_t
        
        self.last_query = text
        self.append_log("Pentagon", f"Mapped to: {domain}\n(Time: {inf_time:.4f} sec)", color="#55ff55")
        
    def process_learning(self, correct_text):
        self.is_learning_mode = False
        target_idx, target_name = self.engine.inference(correct_text, use_thermal=False)
        self.append_log("Pentagon", f"Ah, it belongs to '{target_name}'. Updating landscape...", color="#ffaa00")
        
        self.engine.learn_from_feedback(self.last_query, target_idx)
        self.engine.save_landscape(MODEL_PATH)
        
        self.append_log("System", "Brain updated and saved successfully.", color="#50b3a2")

if __name__ == "__main__":
    app = PentagonGUI()
    app.mainloop()
