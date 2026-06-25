"""Evaluate AuraXR SDF-LSTM checkpoints on temporal HDF5 data."""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mano_fk import MANOForwardKinematics
from model import SDFLSTMModel


def procrustes_align(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    mu_p = pred.mean(0)
    mu_t = tgt.mean(0)
    p_c = pred - mu_p
    t_c = tgt - mu_t
    sigma_p = (p_c ** 2).sum() / len(pred)
    H = p_c.T @ t_c / len(pred)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = S.sum() / (sigma_p + 1e-8)
    return scale * (p_c @ R.T) + mu_t


def pck_auc(errors: np.ndarray, max_thresh: float = 50.0, steps: int = 50) -> float:
    thresholds = np.linspace(0, max_thresh, steps)
    pck = [(errors < t).mean() for t in thresholds]
    return float(np.trapezoid(pck, thresholds) / max_thresh)


@torch.no_grad()
def evaluate_lstm(
    model: SDFLSTMModel,
    h5_path: Path,
    embed_matrix: np.ndarray,
    bop_ids: np.ndarray,
    norm_stats: dict,
    mano_fk,
    device,
    split: str = "val",
) -> dict:
    bop_id_to_idx = {int(bid): i for i, bid in enumerate(bop_ids)}
    emb = torch.from_numpy(embed_matrix.astype(np.float32)).to(device)

    feat_mean = torch.tensor(norm_stats["feature_mean"], dtype=torch.float32).to(device)
    feat_std = torch.tensor(norm_stats["feature_std"], dtype=torch.float32).to(device)
    sdf_mean = torch.tensor(norm_stats["sdf_mean"], dtype=torch.float32).to(device)
    sdf_std = torch.tensor(norm_stats["sdf_std"], dtype=torch.float32).to(device)
    tgt_mean = torch.tensor(norm_stats["target_mean"], dtype=torch.float32).to(device)
    tgt_std = torch.tensor(norm_stats["target_std"], dtype=torch.float32).to(device)

    with h5py.File(h5_path, "r") as f:
        g = f[split]
        features = g["features"][:]
        sdf_feats = g["sdf_features"][:]
        targets = g["targets"][:]
        seq_ids = g["sequence_id"][:]
        frame_idxs = g["frame_index"][:]
        is_mirror = g["is_mirror"][:] if "is_mirror" in g else np.zeros_like(seq_ids)
        obj_ids = g["obj_id"][:]
        distances = g["distances"][:]
        contact_name = "contact_v2" if "contact_v2" in g else "contact"
        contact = g[contact_name][:]

    seq_to_pos: dict[tuple[int, int], list[int]] = {}
    for i in range(len(seq_ids)):
        key = (int(seq_ids[i]), int(is_mirror[i]))
        seq_to_pos.setdefault(key, []).append(i)
    for positions in seq_to_pos.values():
        positions.sort(key=lambda i: frame_idxs[i])

    all_mpjpe: list[float] = []
    all_pa_mpjpe: list[float] = []
    all_mae: list[float] = []
    approach_mpjpe: list[float] = []
    accel_mean: list[float] = []
    contact_correct = 0
    contact_total = 0

    model.eval()
    print(f"Evaluating {len(seq_to_pos)} sequences from {h5_path} [{split}]")

    for seq_i, positions in enumerate(seq_to_pos.values(), start=1):
        if seq_i % 20 == 0:
            print(f"  {seq_i}/{len(seq_to_pos)}")

        feat_np = features[positions]
        sdf_np = sdf_feats[positions]
        tgt_np = targets[positions]
        oid_np = obj_ids[positions]
        dist_np = distances[positions]
        cont_np = contact[positions]

        feat_t = (torch.from_numpy(feat_np.astype(np.float32)).to(device) - feat_mean) / feat_std
        sdf_t = (torch.from_numpy(sdf_np.astype(np.float32)).to(device) - sdf_mean) / sdf_std
        inp_t = torch.cat([feat_t, sdf_t], dim=-1)

        h, c = model.initial_state(batch_size=1, device=device)
        pred_angles = []
        for t in range(len(positions)):
            emb_idx = bop_id_to_idx.get(int(oid_np[t]), 0)
            obj_emb = emb[emb_idx].unsqueeze(0)
            pose, _, _, h, c = model(inp_t[t:t + 1], obj_emb, h, c)
            pred_angles.append(pose.squeeze(0).cpu().numpy())

        pred_raw = np.stack(pred_angles) * tgt_std.cpu().numpy() + tgt_mean.cpu().numpy()
        tgt_raw = tgt_np
        wrist_dummy = np.tile([1, 0, 0, 0, 1, 0], (len(pred_raw), 1)).astype(np.float32)

        if mano_fk is not None:
            pred_pos = mano_fk(pred_raw, wrist_dummy) * 1000.0
            tgt_pos = mano_fk(tgt_raw, wrist_dummy) * 1000.0
        else:
            pred_pos = tgt_pos = None

        for i in range(len(pred_raw)):
            mae = np.abs(pred_raw[i] - tgt_raw[i]).mean()
            all_mae.append(float(mae))

            if pred_pos is not None:
                jerr = float(np.linalg.norm(pred_pos[i] - tgt_pos[i], axis=-1).mean())
                pa_pred = procrustes_align(pred_pos[i], tgt_pos[i])
                pa_err = float(np.linalg.norm(pa_pred - tgt_pos[i], axis=-1).mean())
            else:
                jerr = float(np.degrees(mae))
                pa_err = jerr

            all_mpjpe.append(jerr)
            all_pa_mpjpe.append(pa_err)
            if dist_np[i] < 0.20:
                approach_mpjpe.append(jerr)
            if int(cont_np[i]) == 1:
                contact_total += 1
                contact_correct += int(jerr < 5.0)

        if len(pred_raw) >= 3:
            accel_mean.append(float(np.abs(np.diff(pred_raw, n=2, axis=0)).mean()))

    errors = np.array(all_mpjpe)
    return {
        "split": split,
        "n_frames": int(len(all_mpjpe)),
        "mpjpe_mm": float(np.mean(all_mpjpe)),
        "pa_mpjpe_mm": float(np.mean(all_pa_mpjpe)),
        "mae_deg": float(np.degrees(np.mean(all_mae))),
        "pck_auc_50": pck_auc(errors),
        "approach_mpjpe_mm": float(np.mean(approach_mpjpe)) if approach_mpjpe else None,
        "contact_accuracy": float(contact_correct / contact_total) if contact_total else None,
        "smoothness_accel_mean": float(np.mean(accel_mean)) if accel_mean else None,
        "model": "B3_SDF_LSTM",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="Directory containing dataset_mano.h5")
    parser.add_argument("--hand", required=True, choices=["left", "right"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    h5_path = args.data / "dataset_mano.h5"
    with h5py.File(h5_path, "r") as f:
        norm_stats = json.loads(f.attrs["meta"])

    try:
        mano_fk = MANOForwardKinematics(hand=args.hand)
        mano_fk.model
        print(f"Loaded MANO FK ({args.hand})")
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}; falling back to angle error proxy")
        mano_fk = None

    model = SDFLSTMModel().to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"Checkpoint: epoch={ckpt.get('epoch', '?')}")

    embed_matrix = np.load("data/models/sdf_grids/sdf_embed_matrix.npy")
    bop_ids = np.load("data/models/sdf_grids/sdf_bop_ids.npy")

    start = time.time()
    results = evaluate_lstm(model, h5_path, embed_matrix, bop_ids, norm_stats, mano_fk, device, args.split)
    results["inference_time_s"] = time.time() - start

    print(f"MPJPE: {results['mpjpe_mm']:.2f} mm")
    print(f"PA-MPJPE: {results['pa_mpjpe_mm']:.2f} mm")
    print(f"MAE: {results['mae_deg']:.2f} deg")
    print(f"PCK-AUC@50: {results['pck_auc_50']:.3f}")
    if results["contact_accuracy"] is not None:
        print(f"Contact@5mm: {results['contact_accuracy'] * 100:.1f}%")

    out = args.out or Path(f"results/eval_lstm_{args.hand}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"B3_SDF_LSTM": results}, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
