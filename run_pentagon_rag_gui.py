import customtkinter as ctk
import threading
from pentagon_rag_core import PentagonRAGCore

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PentagonRAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pentagon RAG Translator")
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
        
        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Ask about anything...", height=40, font=("Yu Gothic UI", 14))
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", lambda e: self.send_message())
        
        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=80, height=40, font=("Yu Gothic UI", 14, "bold"), command=self.send_message)
        self.send_button.grid(row=0, column=1)
        
        self.core = PentagonRAGCore()
        
        self.add_message("System", "Loading Pentagon RAG Brain (FAISS & Decoder)... Please wait.", "system")
        threading.Thread(target=self.load_core, daemon=True).start()

    def load_core(self):
        try:
            self.core.load()
            self.add_message("System", "Pentagon Translator is Ready!\nIf GEMINI_API_KEY is set, it will generate natural language. Otherwise, it will show raw knowledge fragments.", "system")
        except Exception as e:
            self.add_message("System", f"Failed to load: {e}", "system")

    def add_message(self, sender, text, msg_type="user"):
        color = "transparent"
        text_color = "white"
        
        if msg_type == "user":
            color = "#2b2b2b"
        elif msg_type == "pentagon":
            color = "#1e3d59"
        elif msg_type == "system":
            color = "#4a4a4a"
            text_color = "#cccccc"
            
        msg_frame = ctk.CTkFrame(self.chat_frame, fg_color=color, corner_radius=8)
        msg_frame.pack(fill="x", padx=10, pady=5)
        
        label = ctk.CTkLabel(msg_frame, text=f"[{sender}]\n{text}", justify="left", anchor="w", 
                             font=("Yu Gothic UI", 13), text_color=text_color, wraplength=800)
        label.pack(padx=15, pady=10, fill="x")
        
        # Auto scroll
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
            response, fragments = self.core.query(user_text)
            self.add_message("Pentagon Decoder", response, "pentagon")
        except Exception as e:
            self.add_message("System Error", str(e), "system")
        finally:
            self.send_button.configure(state="normal")

if __name__ == "__main__":
    app = PentagonRAGApp()
    app.mainloop()
