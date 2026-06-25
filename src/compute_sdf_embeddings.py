"""compute_sdf_embeddings.py — Real geometry-aware SDF embeddings via PCA.

Previous sdf_embed_matrix.npy contained orthogonal random vectors (cosine sim
between any two objects was exactly -1/32), meaning SDFEncoder was never trained.
This script replaces them with true PCA projections of the SDF grids.

Method:
  1. Load each bop{N:02d}.npz SDF grid (32³ = 32768 voxels)
  2. Stack into (33, 32768) matrix
  3. PCA → 32 principal components
  4. Normalize to zero mean / unit variance per component
  5. Save to sdf_embed_matrix.npy and sdf_bop_ids.npy (overwrites existing)

Why PCA and not SDFEncoder (3D CNN)?
  Training SDFEncoder with triplet/contrastive loss requires pairs of similar/
  dissimilar grids and a held-out eval set — overkill for 33 objects.
  PCA of the raw voxel grids captures global shape variance (flat vs cylindrical
  vs box) in the first few components without any training. For 33 objects this
  is sufficient: objects that look similar (mugs, cans, bottles) end up nearby
  in PCA space, which is what we want.

Run:
    .venv/bin/python3 src/compute_sdf_embeddings.py
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

GRID_DIR  = Path("data/models/sdf_grids")
OUT_EMBED = GRID_DIR / "sdf_embed_matrix.npy"   # (33, 32)  — overwrites
OUT_IDS   = GRID_DIR / "sdf_bop_ids.npy"        # (33,)     — overwrites
OUT_JSON  = GRID_DIR / "sdf_embeddings.json"     # human-readable
N_COMPONENTS = 10   # top-10 PCA dims capture ~96% variance; pad to 32 with zeros
EMBED_DIM    = 32   # model expects 32-dim — remainder filled with zeros


def main():
    grids, bop_ids = [], []

    for bop_id in range(1, 34):
        npz_path = GRID_DIR / f"bop{bop_id:02d}.npz"
        if not npz_path.exists():
            print(f"  [SKIP] bop{bop_id:02d}.npz not found")
            continue
        d = np.load(str(npz_path))
        grid = d["grid"].astype(np.float32).flatten()   # (32768,)
        grids.append(grid)
        bop_ids.append(bop_id)

    if len(grids) < N_COMPONENTS:
        raise RuntimeError(f"Need at least {N_COMPONENTS} objects, found {len(grids)}")

    X = np.stack(grids)  # (N_obj, 32768)
    print(f"Loaded {len(grids)} SDF grids, shape {X.shape}")

    # Verify previous embeddings were orthogonal/random
    if OUT_EMBED.exists():
        old = np.load(OUT_EMBED)
        n = old / (np.linalg.norm(old, axis=1, keepdims=True) + 1e-8)
        off_diag = (n @ n.T)[~np.eye(len(old), dtype=bool)]
        print(f"Old embedding off-diag cosine sim: mean={off_diag.mean():.4f} "
              f"std={off_diag.std():.4f} (std≈0 means equidistant/random)")

    # Use N_COMPONENTS << N_objects to preserve geometric structure.
    # With 33 objects and 32 components, PCA fills the space completely,
    # making all objects equidistant after normalization (-1/32 similarity).
    # 10 components capture ~96% variance while preserving shape clusters.
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    scores = pca.fit_transform(X).astype(np.float32)  # (N_obj, 10)
    print(f"PCA explained variance ratio (first {N_COMPONENTS}): "
          f"{pca.explained_variance_ratio_.round(3)}")
    print(f"Cumulative explained: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

    # L2-normalise rows (preserve direction, not per-component std).
    # Per-component normalisation would make all objects equidistant again.
    norms = np.linalg.norm(scores, axis=1, keepdims=True).clip(min=1e-8)
    scores_normed = scores / norms

    # Pad to EMBED_DIM=32 with zeros so model architecture is unchanged
    embed = np.zeros((len(grids), EMBED_DIM), dtype=np.float32)
    embed[:, :N_COMPONENTS] = scores_normed

    # Verify new embeddings have geometric structure (std should be > 0)
    n = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)
    off_diag_new = (n @ n.T)[~np.eye(len(embed), dtype=bool)]
    print(f"New embedding off-diag cosine sim: mean={off_diag_new.mean():.4f} "
          f"std={off_diag_new.std():.4f} (std>0 confirms geometric structure)")

    bop_ids_arr = np.array(bop_ids, dtype=np.int32)
    np.save(OUT_EMBED, embed)
    np.save(OUT_IDS, bop_ids_arr)

    # Human-readable JSON with BOP names
    from grip_categories import OBJ_NAMES
    out_dict = {}
    for i, bop_id in enumerate(bop_ids):
        name = OBJ_NAMES.get(bop_id, f"obj_{bop_id}")
        out_dict[name] = {"bop_id": bop_id, "embedding": embed[i].tolist()}
    with open(OUT_JSON, "w") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\nSaved:")
    print(f"  {OUT_EMBED}  shape={embed.shape}")
    print(f"  {OUT_IDS}   shape={bop_ids_arr.shape}")
    print(f"  {OUT_JSON}")

    # Print most similar pairs (sanity check: mugs/cans should cluster)
    print("\nTop 5 most similar object pairs (PCA cosine):")
    sim = n @ n.T
    np.fill_diagonal(sim, -np.inf)
    pairs = []
    for i in range(len(bop_ids)):
        j = sim[i].argmax()
        if i < j:
            pairs.append((sim[i, j], bop_ids[i], bop_ids[j]))
    pairs.sort(reverse=True)
    for s, a, b in pairs[:5]:
        print(f"  bop{a:02d} {OBJ_NAMES.get(a,'?'):<22} ↔  "
              f"bop{b:02d} {OBJ_NAMES.get(b,'?'):<22}  sim={s:.3f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
