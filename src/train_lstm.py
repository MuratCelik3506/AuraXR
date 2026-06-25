"""Train the AuraXR SDF-LSTM model.

SDF-LSTM training:
  - HOT3D input: temporal windows (T=16), each 29-dim (25 core + 4 SDF local)
  - ARCTIC/DexYCB input: single-frame contact-pose windows (T=1)
  - Object embedding: pre-computed 32-dim PCA embedding (lookup by BOP ID)
  - Loss: Huber(mano_pose_15) + 0.3×MSE(wrist_rot) + 0.1×BCE(contact)
  - Multi-source: HOT3D (full dynamics) + ARCTIC + DexYCB (contact augmentation)

Run (single source, backward-compatible):
    .venv/bin/python3 src/train_lstm.py \\
        --data_dir data/processed/hot3d_mano/right/ \\
        --output_dir checkpoints/lstm_right/

Run (multi-source):
    .venv/bin/python3 src/train_lstm.py \\
        --hot3d_dir  data/processed/hot3d_mano/right \\
        --arctic_dir data/processed/arctic_mano/right \\
        --dexycb_dir data/processed/dexycb_mano/right \\
        --output_dir checkpoints/lstm_right/
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model import SDFLSTMModel

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_T      = 16   # ~0.53s at HOT3D 30fps; covers one full approach gesture
WINDOW_STRIDE = 4    # 75% overlap; improves gradient coverage near window edges
BATCH_SIZE    = 256
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 200
PATIENCE      = 50

POSE_LOSS_W    = 1.0
WRIST_LOSS_W   = 0.3
CONTACT_LOSS_W = 0.1

WRIST_DIMS = slice(11, 17)
SS_START_EPOCH = 10
SS_END_EPOCH = 80
SS_MAX_PROB = 0.50


@dataclass(frozen=True)
class SourceConfig:
    name: str
    window_t: int
    train_fraction: float
    loss_weight: float
    contact_loss_w: float


SOURCE_CONFIGS = {
    # HOT3D is the only source with real approach dynamics.
    "hot3d":  SourceConfig("hot3d",  WINDOW_T, 0.70, 1.00, CONTACT_LOSS_W),
    # ARCTIC/DexYCB are contact-pose augmenters: single-frame only, no contact BCE.
    "arctic": SourceConfig("arctic", 1,        0.25, 0.25, 0.0),
    "dexycb": SourceConfig("dexycb", 1,        0.05, 0.50, 0.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TemporalWindowDataset(Dataset):
    """Sliding windows over temporal sequences from dataset_mano.h5.

    norm_stats: if provided externally (e.g. HOT3D master stats), those are used
    instead of the file's own meta. Allows multi-source training on a common scale.
    """

    def __init__(
        self,
        h5_path: Path,
        split: str,
        embed_matrix: np.ndarray,   # (N_obj, 32) SDF embeddings indexed by bop_ids
        bop_ids: np.ndarray,         # (N_obj,) sorted BOP IDs
        window_t: int = WINDOW_T,
        stride: int = WINDOW_STRIDE,
        norm_stats: dict | None = None,
    ):
        super().__init__()
        self.window_t = window_t
        self.embed_matrix = torch.from_numpy(embed_matrix.astype(np.float32))
        self.bop_id_to_idx = {int(bid): i for i, bid in enumerate(bop_ids)}

        with h5py.File(h5_path, "r") as f:
            g = f[split]
            self.features    = torch.from_numpy(g["features"][:])
            self.sdf_feats   = torch.from_numpy(g["sdf_features"][:])
            self.obj_ids     = g["obj_id"][:]
            self.targets     = torch.from_numpy(g["targets"][:])
            self.wrist_rot   = torch.from_numpy(g["wrist_rot_6d"][:])
            self.seq_ids     = g["sequence_id"][:]
            self.frame_idx   = g["frame_index"][:]
            self.is_mirror   = g["is_mirror"][:] if "is_mirror" in g else np.zeros_like(self.seq_ids)
            contact_name     = "contact_v2" if "contact_v2" in g else "contact"
            self.contact     = torch.from_numpy(g[contact_name][:]).float()

            # Use provided norm_stats or fall back to file's embedded meta
            if norm_stats is None:
                if "meta" not in f.attrs:
                    raise KeyError(
                        f"{h5_path} has no 'meta' attribute. "
                        "Re-run the builder to regenerate, or pass norm_stats explicitly."
                    )
                norm_stats = json.loads(f.attrs["meta"])

        feat_mean = torch.tensor(norm_stats["feature_mean"], dtype=torch.float32)
        feat_std  = torch.tensor(norm_stats["feature_std"],  dtype=torch.float32)
        sdf_mean  = torch.tensor(norm_stats["sdf_mean"],     dtype=torch.float32)
        sdf_std   = torch.tensor(norm_stats["sdf_std"],      dtype=torch.float32)
        tgt_mean  = torch.tensor(norm_stats["target_mean"],  dtype=torch.float32)
        tgt_std   = torch.tensor(norm_stats["target_std"],   dtype=torch.float32)

        self.features  = (self.features  - feat_mean) / feat_std
        self.sdf_feats = (self.sdf_feats - sdf_mean)  / sdf_std
        self.targets   = (self.targets   - tgt_mean)  / tgt_std
        self.tgt_mean = tgt_mean
        self.tgt_std  = tgt_std
        self.wrist_input_mean = feat_mean[WRIST_DIMS]
        self.wrist_input_std = feat_std[WRIST_DIMS]

        # Build sliding window index grouped by (sequence_id, is_mirror)
        self._seq_windows: list[list[int]] = []
        seq_to_frames: dict[tuple[int, int], list[int]] = {}
        for i in range(len(self.seq_ids)):
            key = (int(self.seq_ids[i]), int(self.is_mirror[i]))
            seq_to_frames.setdefault(key, []).append(i)

        for key, positions in seq_to_frames.items():
            positions.sort(key=lambda i: self.frame_idx[i])
            L = len(positions)
            for start in range(0, L - window_t + 1, stride):
                self._seq_windows.append(positions[start:start + window_t])

    def __len__(self):
        return len(self._seq_windows)

    def __getitem__(self, idx: int):
        positions = self._seq_windows[idx]
        feat_seq  = self.features[positions]
        sdf_seq   = self.sdf_feats[positions]
        inp_seq   = torch.cat([feat_seq, sdf_seq], dim=-1)   # (T, 29)
        tgt_seq   = self.targets[positions]
        wrist_seq = self.wrist_rot[positions]
        cont_seq  = self.contact[positions]

        bop_id    = int(self.obj_ids[positions[-1]])
        embed_idx = self.bop_id_to_idx.get(bop_id, 0)
        obj_emb   = self.embed_matrix[embed_idx]              # (32,)

        return inp_seq, obj_emb, tgt_seq, wrist_seq, cont_seq


def load_norm_stats(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        return json.loads(f.attrs["meta"])


def build_multi_source_datasets(
    args,
    embed_matrix: np.ndarray,
    bop_ids: np.ndarray,
) -> tuple[Dataset, list[tuple[str, Dataset]]]:
    """Build train and validation datasets for HOT3D plus optional sources."""
    train_sources: list[tuple[SourceConfig, Dataset]] = []
    val_sources: list[tuple[str, Dataset]] = []

    hot3d_h5 = Path(args.hot3d_dir) / "dataset_mano.h5"
    if not hot3d_h5.exists():
        raise FileNotFoundError(f"HOT3D h5 not found: {hot3d_h5}")

    master_norm = load_norm_stats(hot3d_h5)

    cfg = SOURCE_CONFIGS["hot3d"]
    hot3d_train = TemporalWindowDataset(hot3d_h5, "train", embed_matrix, bop_ids,
                                         window_t=cfg.window_t, stride=WINDOW_STRIDE,
                                         norm_stats=master_norm)
    hot3d_val   = TemporalWindowDataset(hot3d_h5, "val",   embed_matrix, bop_ids,
                                         window_t=cfg.window_t, stride=WINDOW_STRIDE,
                                         norm_stats=master_norm)
    train_sources.append((cfg, hot3d_train))
    val_sources.append(("hot3d", hot3d_val))
    print(f"HOT3D  train={len(hot3d_train)} windows  val={len(hot3d_val)} windows")

    if getattr(args, "arctic_dir", None):
        arctic_h5 = Path(args.arctic_dir) / "dataset_mano.h5"
        if arctic_h5.exists():
            cfg = SOURCE_CONFIGS["arctic"]
            arctic_train = TemporalWindowDataset(arctic_h5, "train", embed_matrix, bop_ids,
                                                  window_t=cfg.window_t, stride=1,
                                                  norm_stats=master_norm)
            arctic_val   = TemporalWindowDataset(arctic_h5, "val",   embed_matrix, bop_ids,
                                                  window_t=cfg.window_t, stride=1,
                                                  norm_stats=master_norm)
            train_sources.append((cfg, arctic_train))
            val_sources.append(("arctic", arctic_val))
            print(f"ARCTIC train={len(arctic_train)} windows  val={len(arctic_val)} windows")
        else:
            print(f"[SKIP] ARCTIC h5 not found: {arctic_h5}")

    if getattr(args, "dexycb_dir", None):
        dexycb_h5 = Path(args.dexycb_dir) / "dataset_mano.h5"
        if dexycb_h5.exists():
            cfg = SOURCE_CONFIGS["dexycb"]
            dexycb_train = TemporalWindowDataset(dexycb_h5, "train", embed_matrix, bop_ids,
                                                  window_t=cfg.window_t, stride=1,
                                                  norm_stats=master_norm)
            dexycb_val   = TemporalWindowDataset(dexycb_h5, "val",   embed_matrix, bop_ids,
                                                  window_t=cfg.window_t, stride=1,
                                                  norm_stats=master_norm)
            train_sources.append((cfg, dexycb_train))
            val_sources.append(("dexycb", dexycb_val))
            print(f"DexYCB train={len(dexycb_train)} windows  val={len(dexycb_val)} windows")
        else:
            print(f"[SKIP] DexYCB h5 not found: {dexycb_h5}")

    return train_sources, val_sources


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def temporal_loss(
    pred_pose,
    pred_wrist,
    pred_contact,
    tgt_pose,
    tgt_wrist,
    contact_label,
    contact_loss_w: float = CONTACT_LOSS_W,
    total_weight: float = 1.0,
):
    pose_loss    = F.huber_loss(pred_pose, tgt_pose, delta=0.5)
    wrist_loss   = F.mse_loss(pred_wrist, tgt_wrist)
    if contact_loss_w > 0:
        contact_loss = F.binary_cross_entropy(pred_contact.squeeze(-1), contact_label)
    else:
        contact_loss = pred_contact.sum() * 0.0
    total = (POSE_LOSS_W * pose_loss
           + WRIST_LOSS_W * wrist_loss
           + contact_loss_w * contact_loss)
    total = total * total_weight
    return total, pose_loss.detach(), wrist_loss.detach(), contact_loss.detach()


def evaluate_temporal(model, loader, device, contact_loss_w: float = CONTACT_LOSS_W) -> tuple[float, float]:
    model.eval()
    total_loss = total_pose_err = n = 0
    with torch.no_grad():
        for inp_seq, obj_emb, tgt_seq, wrist_seq, cont_seq in loader:
            inp_seq   = inp_seq.to(device)
            obj_emb   = obj_emb.to(device)
            tgt_seq   = tgt_seq.to(device)
            wrist_seq = wrist_seq.to(device)
            cont_seq  = cont_seq.to(device)
            pp, pw, pc = model.forward_sequence(inp_seq, obj_emb)
            loss, *_ = temporal_loss(pp, pw, pc, tgt_seq, wrist_seq, cont_seq,
                                     contact_loss_w=contact_loss_w)
            total_loss += loss.item() * inp_seq.shape[0]
            total_pose_err += (pp[:, -1, :] - tgt_seq[:, -1, :]).abs().mean().item() * inp_seq.shape[0]
            n += inp_seq.shape[0]
    return total_loss / n, total_pose_err / n


def _next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def build_source_schedule(train_loaders: dict[str, DataLoader]) -> list[str]:
    """Create one epoch schedule with HOT3D as the anchor and aux sources sampled by ratio."""
    if "hot3d" not in train_loaders:
        return [name for name, loader in train_loaders.items() for _ in range(len(loader))]

    hot3d_steps = len(train_loaders["hot3d"])
    hot3d_fraction = SOURCE_CONFIGS["hot3d"].train_fraction
    schedule = ["hot3d"] * hot3d_steps

    for name, loader in train_loaders.items():
        if name == "hot3d":
            continue
        cfg = SOURCE_CONFIGS[name]
        # Small contact-pose datasets are intentionally oversampled by cycling
        # their loader, so their configured fraction is preserved.
        steps = int(round(hot3d_steps * cfg.train_fraction / hot3d_fraction))
        schedule.extend([name] * steps)

    rng = np.random.default_rng()
    rng.shuffle(schedule)
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Main training loops
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    epochs     = getattr(args, "epochs",     EPOCHS)
    patience   = getattr(args, "patience",   PATIENCE)
    batch_size = getattr(args, "batch_size", BATCH_SIZE)
    print(f"Training SDF-LSTM on {device}")

    embed_matrix = np.load("data/models/sdf_grids/sdf_embed_matrix.npy")
    bop_ids      = np.load("data/models/sdf_grids/sdf_bop_ids.npy")

    use_multi = bool(getattr(args, "hot3d_dir", None))
    if use_multi:
        train_sources, val_sources = build_multi_source_datasets(args, embed_matrix, bop_ids)
        print("Train sources: " + ", ".join(
            f"{cfg.name}=windows:{len(ds)} T:{cfg.window_t} loss_w:{cfg.loss_weight} contact_w:{cfg.contact_loss_w}"
            for cfg, ds in train_sources
        ))
    else:
        h5_path = Path(args.data_dir) / "dataset_mano.h5"
        train_ds = TemporalWindowDataset(h5_path, "train", embed_matrix, bop_ids)
        val_ds   = TemporalWindowDataset(h5_path, "val",   embed_matrix, bop_ids)
        val_sources = [("val", val_ds)]
        print(f"Train: {len(train_ds)} windows  Val: {len(val_ds)} windows")

    if use_multi:
        train_loaders = {
            cfg.name: DataLoader(ds, batch_size=batch_size, shuffle=True,
                                 num_workers=4, persistent_workers=True, pin_memory=False)
            for cfg, ds in train_sources
        }
        source_cfgs = {cfg.name: cfg for cfg, _ in train_sources}
        train_loader = None
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, persistent_workers=True, pin_memory=False)
        train_loaders = {}
        source_cfgs = {}
    val_loaders  = [(name, DataLoader(ds, batch_size=batch_size, shuffle=False,
                                      num_workers=2, persistent_workers=True))
                    for name, ds in val_sources]

    model = SDFLSTMModel().to(device)
    print(f"SDFLSTMModel params: {model.count_params():,}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception:
        print("  torch.compile: skipped (MPS fallback)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=5e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=5)

    best_mpjpe = float("inf")
    patience_counter = 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = n = 0
        if use_multi:
            schedule = build_source_schedule(train_loaders)
            loader_iters = {name: iter(loader) for name, loader in train_loaders.items()}
            batch_iter = tqdm(schedule, desc=f"Epoch {epoch}/{epochs}", leave=False)
        else:
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        source_seen: dict[str, int] = {}
        for item in batch_iter:
            if use_multi:
                source_name = item
                batch, loader_iters[source_name] = _next_batch(
                    loader_iters[source_name], train_loaders[source_name])
                inp_seq, obj_emb, tgt_seq, wrist_seq, cont_seq = batch
                cfg = source_cfgs[source_name]
                source_seen[source_name] = source_seen.get(source_name, 0) + inp_seq.shape[0]
            else:
                source_name = "single"
                cfg = SourceConfig("single", WINDOW_T, 1.0, 1.0, CONTACT_LOSS_W)
                inp_seq, obj_emb, tgt_seq, wrist_seq, cont_seq = item

            inp_seq   = inp_seq.to(device, non_blocking=True)
            obj_emb   = obj_emb.to(device, non_blocking=True)
            tgt_seq   = tgt_seq.to(device, non_blocking=True)
            wrist_seq = wrist_seq.to(device, non_blocking=True)
            cont_seq  = cont_seq.to(device, non_blocking=True)

            pj, pw, pc = model.forward_sequence(inp_seq, obj_emb)
            loss, *_ = temporal_loss(pj, pw, pc, tgt_seq, wrist_seq, cont_seq,
                                     contact_loss_w=cfg.contact_loss_w,
                                     total_weight=cfg.loss_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * inp_seq.shape[0]
            n += inp_seq.shape[0]

        train_loss = total_loss / n

        # Evaluate each val source separately
        val_results = {}
        for name, vloader in val_loaders:
            contact_w = SOURCE_CONFIGS.get(name, SOURCE_CONFIGS["hot3d"]).contact_loss_w
            vloss, vmpjpe = evaluate_temporal(model, vloader, device, contact_loss_w=contact_w)
            val_results[name] = (vloss, vmpjpe)

        # Primary metric: HOT3D val loss (or first source if HOT3D not named)
        primary_name = "hot3d" if "hot3d" in val_results else list(val_results.keys())[0]
        primary_loss, primary_mpjpe = val_results[primary_name]

        if epoch <= 5:
            warmup_scheduler.step()
        else:
            scheduler.step(primary_loss)

        val_summary = "  ".join(
            f"{name}_loss={v:.4f} {name}_pose={m:.4f}"
            for name, (v, m) in val_results.items()
        )
        print(f"Epoch {epoch:4d} | train={train_loss:.4f}  {val_summary}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        if use_multi:
            print("  source_samples=" + ", ".join(
                f"{k}:{v}" for k, v in sorted(source_seen.items())
            ))

        if primary_mpjpe < best_mpjpe:
            best_mpjpe = primary_mpjpe
            patience_counter = 0
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"epoch": epoch, "model": raw.state_dict(),
                        "val_results": {k: {"loss": v, "pose_err": m}
                                        for k, (v, m) in val_results.items()}},
                       out_dir / "best.pt")
            print(f"  ✓ Saved best ({primary_name} pose_err={best_mpjpe:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"Training complete. Best {primary_name} val_pose_err: {best_mpjpe:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=Path, default=None,
                   help="Path to directory containing dataset_mano.h5 (single-source mode)")
    p.add_argument("--hot3d_dir",  type=Path, default=None,
                   help="HOT3D processed dir (multi-source mode, provides master norm stats)")
    p.add_argument("--arctic_dir", type=Path, default=None,
                   help="ARCTIC processed dir (optional augmentation)")
    p.add_argument("--dexycb_dir", type=Path, default=None,
                   help="DexYCB processed dir (optional augmentation)")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--hand",       default="shared", choices=["right", "left", "shared", "canonical"])
    p.add_argument("--epochs",     type=int,  default=EPOCHS)
    p.add_argument("--patience",   type=int,  default=PATIENCE)
    p.add_argument("--batch_size", type=int,  default=BATCH_SIZE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.hot3d_dir is None and args.data_dir is None:
        raise SystemExit("Specify either --data_dir (single) or --hot3d_dir (multi-source)")
    train_lstm(args)
