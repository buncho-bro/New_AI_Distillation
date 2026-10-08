import customtkinter as ctk
import threading
from pentagon_fusion_core import PentagonFusionCore

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green") # フュージョン用は緑テーマ

class PentagonFusionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pentagon FUSION (Direct Projection)")
        self.geometry("900x700")
        
        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Main chat frame
        self.chat_frame = ctk.CTkScrollableFrame(self)
        self.chat_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        
        # Input frame
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Enter concepts...", height=40, font=("Yu Gothic UI", 14))
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", lambda e: self.send_message())
        
        self.send_button = ctk.CTkButton(self.input_frame, text="Project", width=80, height=40, font=("Yu Gothic UI", 14, "bold"), command=self.send_message)
        self.send_button.grid(row=0, column=1)
        
        self.core = PentagonFusionCore()
        
        self.add_message("System", "Loading Fusion Brain (Massive FAISS + Neural Decoder)... Please wait.", "system")
        threading.Thread(target=self.load_core, daemon=True).start()

    def load_core(self):
        try:
            self.core.load()
            self.add_message("System", "Fusion System is Ready!\nAll responses are generated directly from vectors via neural network, without using external APIs or LLMs.", "system")
        except Exception as e:
            self.add_message("System", f"Failed to load: {e}", "system")

    def add_message(self, sender, text, msg_type="user", meta=""):
        color = "transparent"
        text_color = "white"
        
        if msg_type == "user":
            color = "#1a1a1a"
        elif msg_type == "pentagon":
            color = "#0a3d2e"
        elif msg_type == "system":
            color = "#333333"
            text_color = "#aaaaaa"
            
        msg_frame = ctk.CTkFrame(self.chat_frame, fg_color=color, corner_radius=8)
        msg_frame.pack(fill="x", padx=10, pady=5)
        
        display_text = f"[{sender}]\n{text}"
        if meta:
            display_text = f"[{sender} | {meta}]\n{text}"
            
        label = ctk.CTkLabel(msg_frame, text=display_text, justify="left", anchor="w", 
                             font=("Yu Gothic UI", 13), text_color=text_color, wraplength=800)
        label.pack(padx=15, pady=10, fill="x")
        
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def send_message(self):
        user_text = self.entry.get().strip()
        if not user_text: return
        
        self.entry.delete(0, 'end')
        self.add_message("You", user_text, "user")
        
        self.send_button.configure(state="disabled")
        threading.Thread(target=self.process_query, args=(user_text,), daemon=True).start()

    def process_query(self, user_text):
        try:
            decoded_text, valley_id, dist = self.core.query(user_text)
            meta = f"Gravity Fall -> Valley #{valley_id} (Dist: {dist:.3f})"
            
            # ニューラルネットの出力は不完全な場合があるので修飾
            final_text = f"« {decoded_text} »"
            
            self.add_message("Neural Projection", final_text, "pentagon", meta=meta)
        except Exception as e:
            self.add_message("System Error", str(e), "system")
        finally:
            self.send_button.configure(state="normal")

if __name__ == "__main__":
    app = PentagonFusionApp()
    app.mainloop()
