"""
14_quick_train_export.py — Fixed, self-contained train + ONNX export for Unity.

WHAT THIS FIXES vs the previous training run:
  1. Drops delta_q and delta_t losses — both are constant placeholder targets
     (identity quaternion / zero vector) so training on them is meaningless.
     The model still PREDICTS delta_t/delta_q but we don't supervise them here.
  2. Uses a compact GRU model (fast to train on CPU/MPS).
  3. Trains on a small subset (--n_samples, default 20000) so this finishes in minutes.
  4. Exports a valid ONNX opset-17 file ready for Unity Sentis.

Output:
  data/checkpoints/quick_best.pt
  data/intentformer.onnx          ← drag into Unity Assets
  data/intentformer_meta.json     ← load at runtime for normalisation

Usage:
    python 14_quick_train_export.py                     # default 20k samples
    python 14_quick_train_export.py --n_samples 5000    # even faster smoke test
    python 14_quick_train_export.py --epochs 50         # more epochs
    python 14_quick_train_export.py --validate          # ONNX runtime check after export
"""

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_FILE = Path("../data/hot3d_training.h5")
CKPT_DIR  = Path("../data/checkpoints")
ONNX_OUT  = Path("../data/intentformer.onnx")
META_OUT  = Path("../data/intentformer_meta.json")

# ---------------------------------------------------------------------------
# Dimensions — must match 09_build_dataset.py / 10_intentformer.py
# ---------------------------------------------------------------------------

F_IN       = 96   # feature dim per frame
T          = 16   # temporal window length
TARGET_DIM = 78   # output dim

# Output layout (indices into the 78-dim output):
#   [0:15]  mano_pose_h0,  [15:25] mano_betas_h0
#   [25:28] wrist_t_h0,    [28:32] wrist_q_h0
#   [32:35] delta_t_h0,    [35:39] delta_q_h0   ← placeholder (not trained)
#   [39:78] same for hand 1

PER_HAND   = 39   # 15+10+3+4+3+4

# ---------------------------------------------------------------------------
# Compact GRU model (fast, ~1 M params, ONNX-friendly)
# ---------------------------------------------------------------------------

class QuickGRU(nn.Module):
    """
    Bidirectional GRU over the T=16 window.
    Input:  [B, T, F_IN]
    Output: [B, TARGET_DIM]
    Quaternion outputs (wrist_q, delta_q) are L2-normalised.
    """

    def __init__(self, f_in: int = F_IN, hidden: int = 256,
                 num_layers: int = 2, target_dim: int = TARGET_DIM,
                 dropout: float = 0.1):
        super().__init__()
        # Input normalisation (learnable)
        self.input_norm = nn.LayerNorm(f_in)

        # Bidirectional GRU
        self.gru = nn.GRU(
            f_in, hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # MLP head
        gru_out_dim = hidden * 2  # bidirectional
        self.head = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Linear(gru_out_dim, gru_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, target_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F_IN]  →  [B, TARGET_DIM]"""
        x = self.input_norm(x)
        out, _ = self.gru(x)           # [B, T, hidden*2]
        raw = self.head(out[:, -1, :]) # [B, TARGET_DIM]

        # Normalise quaternion outputs to unit-norm
        raw = _normalize_quaternions(raw)
        return raw

    def predict_hand_params(self, output: torch.Tensor) -> dict:
        return _split_output(output)


def _normalize_quaternions(raw: torch.Tensor) -> torch.Tensor:
    """
    Normalise wrist_q and delta_q slices to unit quaternions.
    Fully out-of-place (no in-place ops) — safe on MPS and CUDA.
    Layout: [0:28] | wrist_q_h0[28:32] | [32:35] | delta_q_h0[35:39]
            [39:67] | wrist_q_h1[67:71] | [71:74] | delta_q_h1[74:78]
    """
    return torch.cat([
        raw[:, 0:28],
        F.normalize(raw[:, 28:32], dim=-1),   # wrist_q_h0
        raw[:, 32:35],
        F.normalize(raw[:, 35:39], dim=-1),   # delta_q_h0
        raw[:, 39:67],
        F.normalize(raw[:, 67:71], dim=-1),   # wrist_q_h1
        raw[:, 71:74],
        F.normalize(raw[:, 74:78], dim=-1),   # delta_q_h1
    ], dim=-1)


def _split_output(output: torch.Tensor) -> dict:
    """Split [B, 78] output into named tensors."""
    h0 = output[:, :PER_HAND]
    h1 = output[:, PER_HAND:]
    return {
        "mano_pose_h0":  h0[:, 0:15],
        "mano_betas_h0": h0[:, 15:25],
        "wrist_t_h0":    h0[:, 25:28],
        "wrist_q_h0":    h0[:, 28:32],
        "delta_t_h0":    h0[:, 32:35],
        "delta_q_h0":    h0[:, 35:39],
        "mano_pose_h1":  h1[:, 0:15],
        "mano_betas_h1": h1[:, 15:25],
        "wrist_t_h1":    h1[:, 25:28],
        "wrist_q_h1":    h1[:, 28:32],
        "delta_t_h1":    h1[:, 32:35],
        "delta_q_h1":    h1[:, 35:39],
    }


# ---------------------------------------------------------------------------
# Dataset — reads directly from the HDF5 file
# ---------------------------------------------------------------------------

class HOT3DSubset(Dataset):
    """
    Wraps the existing hot3d_training.h5 with optional normalisation.
    """

    def __init__(self, h5_path: Path, split: str = "train",
                 normalise: bool = True):
        self.h5_path = str(h5_path)
        self.split   = split

        # Load normalisation stats from H5 metadata
        with h5py.File(self.h5_path, "r") as hf:
            self._N = hf[split]["features"].shape[0]
            if normalise:
                meta = json.loads(hf.attrs["meta"])
                self.feat_mean = np.array(meta["feature_mean"], np.float32)
                self.feat_std  = np.array(meta["feature_std"],  np.float32)
                self.tgt_mean  = np.array(meta["target_mean"],  np.float32)
                self.tgt_std   = np.array(meta["target_std"],   np.float32)
            else:
                self.feat_mean = None

        self.normalise = normalise

        # Open file handle (will be opened lazily per-worker)
        self._hf   = None
        self._feat = None
        self._tgt  = None

    def _open(self):
        if self._hf is None:
            self._hf   = h5py.File(self.h5_path, "r")
            self._feat = self._hf[self.split]["features"]
            self._tgt  = self._hf[self.split]["targets"]

    def __len__(self):
        return self._N

    def __getitem__(self, idx):
        self._open()
        feat = self._feat[idx].astype(np.float32)   # [T, F_IN]
        tgt  = self._tgt[idx].astype(np.float32)    # [TARGET_DIM]

        if self.normalise:
            feat = (feat - self.feat_mean) / self.feat_std
            tgt  = (tgt  - self.tgt_mean)  / self.tgt_std

        return torch.from_numpy(feat), torch.from_numpy(tgt)


# ---------------------------------------------------------------------------
# FIXED Loss functions — drop broken delta_q/delta_t losses
# ---------------------------------------------------------------------------

def geodesic_quat_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between unit quaternions. Returns mean scalar."""
    pred_q = F.normalize(pred_q, dim=-1)
    gt_q   = F.normalize(gt_q,   dim=-1)
    dot = (pred_q * gt_q).sum(dim=-1).abs().clamp(1e-6, 1 - 1e-6)
    return (2 * torch.acos(dot)).mean()


def compute_loss(pred: torch.Tensor, target: torch.Tensor,
                 model: nn.Module) -> tuple[torch.Tensor, dict]:
    """
    Fixed loss — trains only on the real MANO + wrist outputs.
    delta_t and delta_q are EXCLUDED (they are placeholder identity values in the dataset).

    Loss terms:
      joint_mse    λ=1.0   MANO pose θ for both hands
      beta_mse     λ=0.3   MANO shape β (low weight, mostly prior)
      wrist_t_mse  λ=1.0   Wrist translation
      wrist_q_geo  λ=1.0   Geodesic quaternion loss on wrist orientation
      vel_smooth   λ=0.2   Penalise pose jitter within batch (proxy for temporal smoothness)
    """
    p = model.predict_hand_params(pred)
    g = model.predict_hand_params(target)

    mse = F.mse_loss

    l_joint  = (mse(p["mano_pose_h0"], g["mano_pose_h0"]) +
                mse(p["mano_pose_h1"], g["mano_pose_h1"])) * 0.5

    l_beta   = (mse(p["mano_betas_h0"], g["mano_betas_h0"]) +
                mse(p["mano_betas_h1"], g["mano_betas_h1"])) * 0.5

    l_wrist_t = (mse(p["wrist_t_h0"], g["wrist_t_h0"]) +
                 mse(p["wrist_t_h1"], g["wrist_t_h1"])) * 0.5

    l_wrist_q = (geodesic_quat_loss(p["wrist_q_h0"], g["wrist_q_h0"]) +
                 geodesic_quat_loss(p["wrist_q_h1"], g["wrist_q_h1"])) * 0.5

    # Velocity proxy: penalise large difference between consecutive batch samples
    # (approximate — exact temporal smoothness requires sequence-ordered batches)
    l_vel = (mse(p["mano_pose_h0"][1:], p["mano_pose_h0"][:-1]) +
             mse(p["mano_pose_h1"][1:], p["mano_pose_h1"][:-1])) * 0.5

    total = (1.0 * l_joint +
             0.3 * l_beta  +
             1.0 * l_wrist_t +
             1.0 * l_wrist_q +
             0.2 * l_vel)

    return total, {
        "joint":   l_joint.item(),
        "beta":    l_beta.item(),
        "wrist_t": l_wrist_t.item(),
        "wrist_q": l_wrist_q.item(),
        "vel":     l_vel.item(),
        "total":   total.item(),
    }


def wrist_mpjpe(pred: torch.Tensor, target: torch.Tensor,
                model: nn.Module) -> float:
    """Proxy MPJPE: wrist translation error in mm (both hands average)."""
    p = model.predict_hand_params(pred)
    g = model.predict_hand_params(target)
    err_h0 = (p["wrist_t_h0"] - g["wrist_t_h0"]).norm(dim=-1).mean()
    err_h1 = (p["wrist_t_h1"] - g["wrist_t_h1"]).norm(dim=-1).mean()
    return float(((err_h0 + err_h1) * 0.5).item() * 1000)  # metres → mm


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, opt, device):
    model.train()
    total, n = 0., 0
    for feat, tgt in loader:
        feat, tgt = feat.to(device), tgt.to(device)
        pred = model(feat)
        loss, _ = compute_loss(pred, tgt, model)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, mpjpes, n = 0., [], 0
    for feat, tgt in loader:
        feat, tgt = feat.to(device), tgt.to(device)
        pred = model(feat)
        loss, _ = compute_loss(pred, tgt, model)
        total += loss.item()
        mpjpes.append(wrist_mpjpe(pred, tgt, model))
        n += 1
    return total / max(n, 1), float(np.mean(mpjpes))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, h5_path: Path,
                onnx_path: Path, meta_path: Path, validate: bool):
    model.eval().cpu()

    dummy = torch.zeros(1, T, F_IN)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        opset_version=17,
        input_names=["features"],
        output_names=["pose"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "pose":     {0: "batch_size"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"\n  Exported : {onnx_path}  ({size_mb:.1f} MB)")

    # Companion metadata for Unity runtime
    with h5py.File(str(h5_path), "r") as hf:
        raw_meta = json.loads(hf.attrs.get("meta", "{}"))

    meta = {
        "model_name":  "QuickGRU",
        "input_name":  "features",
        "output_name": "pose",
        "T":           T,
        "feature_dim": F_IN,
        "target_dim":  TARGET_DIM,
        "feature_mean": raw_meta.get("feature_mean", [0.0] * F_IN),
        "feature_std":  raw_meta.get("feature_std",  [1.0] * F_IN),
        "target_mean":  raw_meta.get("target_mean",  [0.0] * TARGET_DIM),
        "target_std":   raw_meta.get("target_std",   [1.0] * TARGET_DIM),
        "output_layout": {
            "mano_pose_h0":  [0,  15],
            "mano_betas_h0": [15, 25],
            "wrist_t_h0":    [25, 28],
            "wrist_q_h0":    [28, 32],
            "delta_t_h0":    [32, 35],
            "delta_q_h0":    [35, 39],
            "mano_pose_h1":  [39, 54],
            "mano_betas_h1": [54, 64],
            "wrist_t_h1":    [64, 67],
            "wrist_q_h1":    [67, 71],
            "delta_t_h1":    [71, 74],
            "delta_q_h1":    [74, 78],
        },
        "notes": (
            "QuickGRU — compact model trained on a small subset for Unity pipeline validation. "
            "delta_t and delta_q are untrained placeholder outputs (always near identity). "
            "Apply de-normalisation: output * target_std + target_mean. "
            "Quaternions must be L2-normalised after de-normalisation."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata : {meta_path}")

    if validate:
        print("\n  Validating ONNX output...")
        try:
            import onnxruntime as ort
            sess     = ort.InferenceSession(str(onnx_path))
            dummy_np = dummy.numpy()
            onnx_out = sess.run(None, {"features": dummy_np})[0]
            with torch.no_grad():
                pt_out = model(dummy).numpy()
            max_diff = float(np.abs(pt_out - onnx_out).max())
            ok = "✓ OK" if max_diff < 1e-4 else "✗ LARGE"
            print(f"    Output shape : {list(onnx_out.shape)}")
            print(f"    PyTorch vs ONNX max diff: {max_diff:.2e}  {ok}")
        except ImportError:
            print("    [INFO] onnxruntime not installed — skipping check.")
            print("    Install: pip install onnxruntime")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=20_000,
                    help="Number of training samples to use (subset for speed)")
    ap.add_argument("--val_samples", type=int, default=4_000,
                    help="Number of validation samples")
    ap.add_argument("--epochs",    type=int,   default=40)
    ap.add_argument("--batch",     type=int,   default=128)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--hidden",    type=int,   default=256)
    ap.add_argument("--patience",  type=int,   default=10)
    ap.add_argument("--device",    type=str,   default="auto")
    ap.add_argument("--validate",  action="store_true",
                    help="Run ONNX Runtime consistency check after export")
    ap.add_argument("--no_export", action="store_true",
                    help="Skip ONNX export (training only)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()         else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    if not DATA_FILE.exists():
        print(f"[ERROR] {DATA_FILE} not found. Run 09_build_dataset.py first.")
        return

    # ── Datasets ──────────────────────────────────────────────────────────
    print(f"\nLoading {args.n_samples:,} train / {args.val_samples:,} val samples ...")
    train_full = HOT3DSubset(DATA_FILE, "train", normalise=True)
    val_full   = HOT3DSubset(DATA_FILE, "val",   normalise=True)

    n_train = min(args.n_samples,   len(train_full))
    n_val   = min(args.val_samples, len(val_full))

    # Random subset — reproducible
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(train_full), n_train, replace=False)
    val_idx   = rng.choice(len(val_full),   n_val,   replace=False)

    train_ds = Subset(train_full, train_idx.tolist())
    val_ds   = Subset(val_full,   val_idx.tolist())

    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # DataLoader — 0 workers for MPS (h5py pickling issue)
    nw = 0 if device.type == "mps" else 4
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────
    model = QuickGRU(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: QuickGRU  ({n_params:,} params)")

    opt       = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ── Training ──────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path    = CKPT_DIR / "quick_best.pt"
    best_mpjpe   = math.inf
    patience_ctr = 0

    print(f"\n{'Epoch':>6}  {'Train':>9}  {'Val':>9}  {'MPJPE(mm)':>11}  {'Time':>6}")
    print("-" * 55)

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss             = train_epoch(model, train_loader, opt, device)
        val_loss, val_mpjpe = eval_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = val_mpjpe < best_mpjpe
        marker  = " ✓" if is_best else ""
        print(f"{epoch+1:>6}  {tr_loss:>9.4f}  {val_loss:>9.4f}  "
              f"{val_mpjpe:>11.2f}{marker}  {elapsed:>5.1f}s")

        if is_best:
            best_mpjpe   = val_mpjpe
            patience_ctr = 0
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "opt":        opt.state_dict(),
                "best_mpjpe": best_mpjpe,
                "args":       vars(args),
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n[Early stop] No improvement for {args.patience} epochs.")
                break

    print(f"\n[DONE] Best val MPJPE: {best_mpjpe:.2f} mm")
    print(f"Checkpoint: {ckpt_path}")

    # ── ONNX Export ───────────────────────────────────────────────────────
    if not args.no_export:
        print("\n── ONNX Export ────────────────────────────────────────────")
        # Reload best checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        export_onnx(model, DATA_FILE, ONNX_OUT, META_OUT, args.validate)

        print(f"\n══ Unity integration ═══════════════════════════════════════")
        print(f"  1. Drag {ONNX_OUT} into Unity Assets folder")
        print(f"  2. Copy {META_OUT} to Unity StreamingAssets/")
        print(f"  3. Assign the model asset to AuraXRInferenceManager.modelAsset")
        print(f"  4. AuraXRMetaLoader will load intentformer_meta.json at runtime")
        print(f"════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
