"""
run_real_text_experiments.py
============================

Step 3: Grounding the High-Dimensional Concept Core in Real World Data.
Connects a pre-trained LLM embedding model (sentence-transformers) to the 64D 
Concept Core, projecting real semantic space into physical attractors.
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DIM = 64
SCALE = 1.0  # Spatial scale to map embeddings onto

# --- Data ---
CATEGORIES = ["Animals", "Vehicles", "Emotions"]

TEXTS_TRAIN = {
    "Animals": [
        "A golden retriever is barking loudly in the yard.",
        "A small kitten is sleeping peacefully on the sofa.",
        "The lion roars loudly in the dry savanna.",
        "A wild wolf is hunting prey in the dark forest.",
        "A cute puppy wags its tail while waiting for food.",
        "A fierce tiger stalks through the dense jungle.",
        "The brown bear catches salmon in the rushing river.",
        "A flock of birds flies south for the winter.",
        "The energetic chihuahua runs around the park.",
        "A curious cat chases a laser pointer."
    ],
    "Vehicles": [
        "The red sports car accelerates quickly on the highway.",
        "A large airplane is flying high above the thick clouds.",
        "The electric commuter train arrives smoothly at the station.",
        "A massive cargo ship slowly crosses the deep ocean.",
        "The loud motorcycle speeds down the narrow street.",
        "A yellow school bus picks up children in the morning.",
        "The helicopter hovers over the tall city buildings.",
        "A mountain bike races down the steep dirt trail.",
        "A delivery truck parks carefully in front of the store.",
        "The high-speed bullet train connects the two major cities."
    ],
    "Emotions": [
        "She felt immense joy and overwhelming happiness today.",
        "He was entirely consumed by deep sadness and heavy tears.",
        "The man expressed burning anger and severe frustration.",
        "A sense of peaceful calm washed over her in the quiet room.",
        "They were gripped by sudden terror and paralyzing fear.",
        "He smiled with pure delight and warm satisfaction.",
        "She sobbed quietly in profound grief and loneliness.",
        "The angry customer shouted with extreme rage and hostility.",
        "I feel incredibly relaxed and completely stress-free.",
        "A wave of devastating anxiety and depression hit him."
    ]
}

TEXTS_TEST = {
    "Animals": "A happy dog catches a frisbee in the park.",
    "Vehicles": "A supersonic jet aircraft broke the sound barrier.",
    "Emotions": "I feel totally devastated, hopeless, and depressed.",
    "Ambiguous (Boundary)": "The frightened dog ran away from the fast car."
}

def _set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def project_embeddings(embeddings: torch.Tensor, pca_model: PCA) -> torch.Tensor:
    """Project 384D embeddings to 64D using fitted PCA and normalize to unit sphere."""
    emb_pca = torch.tensor(pca_model.transform(embeddings.numpy())).float()
    # Pad to DIM
    emb64 = torch.zeros(embeddings.shape[0], DIM)
    emb64[:, :emb_pca.shape[1]] = emb_pca
    # Normalize to sphere
    emb64 = emb64 / emb64.norm(dim=1, keepdim=True)
    return emb64

def experiment_real_texts() -> None:
    _set_seed()
    print("\n" + "="*65)
    print("STEP 3: Real Text Concept Core (Sentence Transformers)")
    print("="*65)
    
    print("  Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    all_train_texts = []
    all_train_labels = []
    
    for label_idx, cat in enumerate(CATEGORIES):
        texts = TEXTS_TRAIN[cat]
        all_train_texts.extend(texts)
        all_train_labels.extend([label_idx] * len(texts))
        
    print("  Computing embeddings for training texts...")
    emb_train_384 = torch.tensor(model.encode(all_train_texts))
    
    # Use PCA for semantic-preserving dimensionality reduction
    n_comp = min(emb_train_384.shape[0], emb_train_384.shape[1], DIM)
    pca_64 = PCA(n_components=n_comp)
    pca_64.fit(emb_train_384.numpy())
    obs = project_embeddings(emb_train_384, pca_64)
    
    # 2. Initialize Concept Core
    # Spherical constraint naturally prevents collapse to the origin.
    # Therefore we don't need Mexican Hat (use_ridge=False).
    # Since max distance on unit sphere is 1.414, sigma=0.5 is a good radius.
    p = HDParams(
        dim=DIM, use_ridge=False, sigma=0.5,
        decay_rate=0.99, prune_thresh=0.01,
        marble_lr=0.5, marble_momentum=0.8, marble_max_v=0.1
    )
    core = HighDimConceptCore(p)
    
    # 3. Autonomous Discovery (Learning Phase)
    print("\n  Phase A: Autonomous Learning (Rain & Erode) ...")
    final_train_drops = core.rain_and_erode(obs, steps=80, depth_gain=0.15)
    
    for _ in range(30):
        core.weather()
        
    stats = core.weight_stats()
    print(f"  Memory bank after learning: {stats['n']} attractors.")
    
    # 4. Zero-shot Inference (Marble Dropping)
    print("\n  Phase B & C: Zero-shot Inference Test ...")
    
    test_results = {}
    test_trajectories = {}
    
    for cat_name, text in TEXTS_TEST.items():
        emb_test_384 = torch.tensor(model.encode([text]))
        test_obs = project_embeddings(emb_test_384, pca_64)
        
        final_pos, path, steps = core.roll_marble(test_obs)
        test_trajectories[cat_name] = {"start": test_obs, "end": final_pos, "path": path}
        
        # Determine which macro-valley it fell into by comparing to converged training points
        # For simplicity, we just find the closest training final position and use its label
        dists = torch.cdist(final_pos.unsqueeze(0), final_train_drops).squeeze(0)
        closest_idx = dists.argmin().item()
        closest_dist = dists[closest_idx].item()
        pred_label = all_train_labels[closest_idx]
        pred_cat = CATEGORIES[pred_label]
        
        print(f"\n  [Test: {cat_name}] '{text}'")
        print(f"    -> Rolled for {steps} steps.")
        print(f"    -> Closest category: {pred_cat} (dist: {closest_dist:.2f})")
        test_results[cat_name] = pred_cat

    # 5. Visualization (PCA)
    print("\n  Generating PCA visualization of Semantic Landscape ...")
    
    # Fit PCA on training initial + final positions to frame the space nicely
    pca = PCA(n_components=2)
    fit_data = torch.cat([obs, final_train_drops]).numpy()
    pca.fit(fit_data)
    
    obs_2d = pca.transform(obs.numpy())
    fin_2d = pca.transform(final_train_drops.numpy())
    cen_2d = pca.transform(core.centers.detach().numpy())
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Background potential
    pad = 2.0
    x_min, x_max = obs_2d[:,0].min()-pad, obs_2d[:,0].max()+pad
    y_min, y_max = obs_2d[:,1].min()-pad, obs_2d[:,1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_64d = pca.inverse_transform(grid_2d)
    
    with torch.no_grad():
        H_grid = core.potential(torch.from_numpy(grid_64d).float()).numpy()
    H_grid = H_grid.reshape(xx.shape)
    
    cf = ax.contourf(xx, yy, H_grid, levels=30, cmap="terrain", alpha=0.7)
    plt.colorbar(cf, ax=ax, label="Potential H(x)")
    
    # Plot initial training data (larger and higher zorder so they are visible over the background)
    colors = ['#FF5722', '#2196F3', '#4CAF50']
    for i in range(3):
        mask = (np.array(all_train_labels) == i)
        ax.scatter(obs_2d[mask, 0], obs_2d[mask, 1], c=colors[i], marker='o', alpha=0.8, 
                   s=60, edgecolors='white', linewidths=0.5, zorder=3,
                   label=f"Init: {CATEGORIES[i]}")
        # Converged stars on top
        ax.scatter(fin_2d[mask, 0], fin_2d[mask, 1], c=colors[i], marker='*', s=150, edgecolors='k', zorder=5,
                   label=f"Converged: {CATEGORIES[i]}")
                   
    # Plot test trajectories
    test_colors = {'Animals': '#FF5722', 'Vehicles': '#2196F3', 'Emotions': '#4CAF50', 'Ambiguous (Boundary)': '#9C27B0'}
    for name, data in test_trajectories.items():
        start_2d = pca.transform(data["start"].numpy())[0]
        end_2d = pca.transform(data["end"].unsqueeze(0).numpy())[0]
        path_tensor = torch.stack(data["path"])
        path_2d = pca.transform(path_tensor.numpy())
        
        # Plot trajectory path
        ax.plot(path_2d[:,0], path_2d[:,1], c=test_colors[name], lw=3, linestyle='-', alpha=0.8, zorder=6)
        # Add an arrow at the end to show direction
        if len(path_2d) > 2:
            ax.annotate("", xy=(path_2d[-1,0], path_2d[-1,1]), xytext=(path_2d[-3,0], path_2d[-3,1]),
                        arrowprops=dict(arrowstyle="->", color=test_colors[name], lw=3), zorder=6)
        
        # Test start
        ax.scatter(start_2d[0], start_2d[1], c=test_colors[name], marker='s', s=120, edgecolors='k', zorder=7, label=f"Test Start ({name})")
        
    ax.set_title("Semantic Concept Landscape (64D Real Text -> 2D PCA)", fontsize=14)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    
    # Custom legend
    handles, labels_leg = ax.get_legend_handles_labels()
    unique = dict(zip(labels_leg, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1.15, 1.0))
    
    out_path = os.path.join(OUTPUT_DIR, "real_text_concept_core.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")

if __name__ == "__main__":
    experiment_real_texts()
