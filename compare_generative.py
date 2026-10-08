import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Import the Concept Core
from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DIM = 64

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class PrefixProjector(nn.Module):
    def __init__(self, dim_in, dim_out, num_tokens=2):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim_out = dim_out
        self.net = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.ReLU(),
            nn.Linear(256, num_tokens * dim_out)
        )
    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.num_tokens, self.dim_out)

def train_projector(projector, gpt_model, tokenizer, core_coords, texts, epochs=150, lr=0.005):
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)
    gpt_model.eval()
    for param in gpt_model.parameters():
        param.requires_grad = False

    print("Training Prefix-Tuning Projector (Mapping Concepts to Language)...")
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(texts)):
            text = texts[i]
            coord = core_coords[i:i+1] # (1, D)
            
            inputs = tokenizer(text, return_tensors="pt")
            target_ids = inputs.input_ids
            
            prefix_embeds = projector(coord) # (1, K, 768)
            word_embeds = gpt_model.transformer.wte(target_ids) # (1, L, 768)
            inputs_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)
            
            outputs = gpt_model(inputs_embeds=inputs_embeds)
            logits = outputs.logits # (1, K+L, V)
            
            shift_logits = logits[0, projector.num_tokens-1 : -1, :].contiguous()
            shift_labels = target_ids[0].contiguous()
            
            loss = nn.CrossEntropyLoss()(shift_logits, shift_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(texts):.4f}")

def roll_marble(core, start_pt):
    x = start_pt.clone().detach().float().requires_grad_(True)
    v = torch.zeros_like(x)
    for _ in range(core.p.marble_steps):
        H = core.potential(x.unsqueeze(0)).squeeze()
        H.backward()
        grad = x.grad.detach()
        v = core.p.marble_momentum * v - core.p.marble_lr * grad
        v_norm = v.norm()
        if v_norm > core.p.marble_max_v:
            v = v * (core.p.marble_max_v / v_norm)
        x_new = x.detach() + v
        x_new = x_new / x_new.norm()
        x = x_new.requires_grad_(True)
        if v.norm() < core.p.marble_vel_thresh:
            break
    return x.detach()

def run_comparison():
    set_seed(42)
    print("Loading Models...")
    # 1. Models
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # 2. Anchor Concepts (The pure attractors)
    concept_texts = [
        # Animals
        "A loyal dog.", "A cute cat.", "A wild lion.", 
        # Vehicles
        "A fast car.", "A flying airplane.", "A large train.",
        # Emotions
        "Feeling very happy.", "Feeling very sad.", "Feeling very angry."
    ]
    
    # Embed & PCA
    print("Creating Concept Core...")
    DIM = 8
    raw_emb = encoder.encode(concept_texts)
    pca = PCA(n_components=DIM, whiten=True)
    pca_emb = torch.tensor(pca.fit_transform(raw_emb)).float()
    pca_emb = pca_emb / pca_emb.norm(dim=1, keepdim=True)
    
    # Setup Physics Core
    p = HDParams(dim=DIM, use_ridge=True, sigma=0.3, sigma_rep=0.45, alpha_rep=0.4, marble_lr=0.5, marble_steps=400)
    core = HighDimConceptCore(p)
    core._centers = pca_emb.clone()
    core._weights = torch.ones(len(concept_texts)) * 0.5
    
    # Train Projector
    projector = PrefixProjector(dim_in=DIM, dim_out=768, num_tokens=3)
    train_projector(projector, gpt_model, tokenizer, pca_emb, concept_texts, epochs=150)
    
    # 3. The Comparison (Ambiguous Prompts)
    test_prompts = [
        "A creature with four legs that barks loudly at strangers.",
        "It has four wheels, an engine, and goes vroom on the highway.",
        "Tears are falling down my face because I lost my favorite toy."
    ]
    
    print("\n=======================================================")
    print(" RAW GPT-2 vs CONCEPT CORE DE-NOISING COMPARISON")
    print("=======================================================\n")
    
    gpt_model.eval()
    
    for prompt in test_prompts:
        print(f"INPUT (Noisy/Ambiguous): \"{prompt}\"")
        
        # --- BASELINE: RAW GPT-2 ---
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = gpt_model.generate(
            inputs.input_ids, 
            max_length=len(inputs.input_ids[0]) + 15, 
            do_sample=True, top_p=0.9, temperature=0.7, pad_token_id=tokenizer.eos_token_id
        )
        raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        raw_generated = raw_output[len(prompt):].strip()
        print(f"  [Raw GPT-2 Output]  : \"{raw_generated}\" (Continues surface text)")
        
        # --- CONCEPT CORE ---
        # 1. Encode
        emb = encoder.encode([prompt])
        emb_pca = torch.tensor(pca.transform(emb)).float()
        emb_pca = emb_pca / emb_pca.norm(dim=1, keepdim=True)
        
        # 2. Roll (Physics De-noising)
        converged_coord = roll_marble(core, emb_pca[0])
        
        # Debug: Check which anchor it converged to
        dists = torch.norm(core._centers - converged_coord.unsqueeze(0), dim=1)
        nearest_idx = torch.argmin(dists).item()
        nearest_dist = dists[nearest_idx].item()
        print(f"  [Physics] Converged to -> '{concept_texts[nearest_idx]}' (Dist: {nearest_dist:.3f})")
        
        # 3. Generate from converged coordinate
        prefix_embeds = projector(converged_coord.unsqueeze(0)) # (1, K, 768)
        generated_ids = []
        past_key_values = None
        
        # First step: pass prefix
        with torch.no_grad():
            outputs = gpt_model(inputs_embeds=prefix_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated_ids.append(next_token.item())
            past_key_values = outputs.past_key_values
            
            # Autoregressive loop
            for _ in range(15):
                out = gpt_model(input_ids=next_token.unsqueeze(0), past_key_values=past_key_values)
                next_token_logits = out.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                if next_token.item() == tokenizer.eos_token_id or next_token.item() == 198: # Stop on newline
                    break
                generated_ids.append(next_token.item())
                past_key_values = out.past_key_values
                
        concept_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"  [Concept Core Output]: \"{concept_output}\" (Extracts pure essence)")
        print("-" * 60)

if __name__ == "__main__":
    run_comparison()
