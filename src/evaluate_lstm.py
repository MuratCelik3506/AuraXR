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

WRIST_DIMS = slice(11, 17)


def infer_feat_dim_from_state_dict(state_dict: dict[str, torch.Tensor], fallback: int) -> int:
    weight = state_dict.get("feat_proj.0.weight")
    return int(weight.shape[1]) if weight is not None else fallback


def infer_orientation_aware_from_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    weight = state_dict.get("obj_inj.0.weight")
    return bool(weight is not None and int(weight.shape[1]) == 99)


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-6)
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_angle_deg(pred_6d: torch.Tensor, tgt_6d: torch.Tensor) -> torch.Tensor:
    r_pred = rot6d_to_matrix(pred_6d)
    r_tgt = rot6d_to_matrix(tgt_6d)
    r_diff = r_pred.transpose(-2, -1) @ r_tgt
    trace = r_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.rad2deg(torch.acos(cos))


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
        wrist_targets = g["wrist_rot_6d"][:]
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
    contact_correct_10 = 0
    contact_correct_15 = 0
    contact_total = 0
    binary_contact_correct = 0
    binary_contact_total = 0
    contact_mpjpe: list[float] = []
    wrist_deg: list[float] = []
    final_wrist_deg: list[float] = []
    contact_wrist_deg: list[float] = []
    wrist_jitter_deg: list[float] = []

    model.eval()
    print(f"Evaluating {len(seq_to_pos)} sequences from {h5_path} [{split}]")

    for seq_i, positions in enumerate(seq_to_pos.values(), start=1):
        if seq_i % 20 == 0:
            print(f"  {seq_i}/{len(seq_to_pos)}")

        feat_np = features[positions]
        sdf_np = sdf_feats[positions]
        tgt_np = targets[positions]
        wrist_np = wrist_targets[positions]
        oid_np = obj_ids[positions]
        dist_np = distances[positions]
        cont_np = contact[positions]

        feat_t = (torch.from_numpy(feat_np.astype(np.float32)).to(device) - feat_mean) / feat_std
        sdf_t = (torch.from_numpy(sdf_np.astype(np.float32)).to(device) - sdf_mean) / sdf_std
        inp_t = torch.cat([feat_t, sdf_t], dim=-1)
        model_feat_dim = model.feat_proj[0].in_features
        if inp_t.shape[1] > model_feat_dim:
            inp_t = inp_t[:, :model_feat_dim]
        elif inp_t.shape[1] < model_feat_dim:
            inp_t = torch.nn.functional.pad(inp_t, (0, model_feat_dim - inp_t.shape[1]))

        h, c = model.initial_state(batch_size=1, device=device)
        pred_angles = []
        pred_wrists = []
        pred_contacts = []
        prev_wrist = None
        for t in range(len(positions)):
            emb_idx = bop_id_to_idx.get(int(oid_np[t]), 0)
            obj_emb = emb[emb_idx].unsqueeze(0)
            frame = inp_t[t:t + 1].clone()
            if prev_wrist is not None and frame.shape[1] >= WRIST_DIMS.stop:
                frame[:, WRIST_DIMS] = (
                    prev_wrist - feat_mean[WRIST_DIMS]
                ) / feat_std[WRIST_DIMS].clamp_min(1e-6)
            pose, wrist, contact_prob, h, c = model(frame, obj_emb, h, c)
            pred_angles.append(pose.squeeze(0).cpu().numpy())
            pred_wrists.append(wrist.squeeze(0).cpu())
            pred_contacts.append(float(contact_prob.item()))
            prev_wrist = wrist.detach()

        pred_raw = np.stack(pred_angles) * tgt_std.cpu().numpy() + tgt_mean.cpu().numpy()
        tgt_raw = tgt_np
        pred_wrist_t = torch.stack(pred_wrists)
        tgt_wrist_t = torch.from_numpy(wrist_np.astype(np.float32))
        wrist_err = rotation_angle_deg(pred_wrist_t, tgt_wrist_t).numpy()
        wrist_deg.extend(float(x) for x in wrist_err)
        final_wrist_deg.append(float(wrist_err[-1]))
        contact_wrist_deg.extend(float(wrist_err[i]) for i, cval in enumerate(cont_np) if int(cval) == 1)
        if len(wrist_err) >= 2:
            wrist_jitter = rotation_angle_deg(pred_wrist_t[1:], pred_wrist_t[:-1]).mean().item()
            wrist_jitter_deg.append(float(wrist_jitter))
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
                contact_mpjpe.append(jerr)
                contact_correct += int(jerr < 5.0)
                contact_correct_10 += int(jerr < 10.0)
                contact_correct_15 += int(jerr < 15.0)
            binary_contact_total += 1
            binary_contact_correct += int((pred_contacts[i] >= 0.5) == (int(cont_np[i]) == 1))

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
        "contact_at_10mm": float(contact_correct_10 / contact_total) if contact_total else None,
        "contact_at_15mm": float(contact_correct_15 / contact_total) if contact_total else None,
        "contact_mpjpe_mm": float(np.mean(contact_mpjpe)) if contact_mpjpe else None,
        "binary_contact_accuracy": float(binary_contact_correct / binary_contact_total) if binary_contact_total else None,
        "wrist_deg": float(np.mean(wrist_deg)) if wrist_deg else None,
        "final_wrist_deg": float(np.mean(final_wrist_deg)) if final_wrist_deg else None,
        "contact_wrist_deg": float(np.mean(contact_wrist_deg)) if contact_wrist_deg else None,
        "wrist_jitter_deg": float(np.mean(wrist_jitter_deg)) if wrist_jitter_deg else None,
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

    ckpt = torch.load(args.ckpt, map_location="cpu")
    fallback_dim = len(norm_stats["feature_mean"]) + len(norm_stats["sdf_mean"])
    feat_dim = infer_feat_dim_from_state_dict(ckpt["model"], fallback_dim)
    orientation_aware = infer_orientation_aware_from_state_dict(ckpt["model"])
    model = SDFLSTMModel(feat_dim=feat_dim, orientation_aware_sdf=orientation_aware).to(device)
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
        print(f"Contact@10mm: {results['contact_at_10mm'] * 100:.1f}%")
        print(f"Contact@15mm: {results['contact_at_15mm'] * 100:.1f}%")
        print(f"Contact MPJPE: {results['contact_mpjpe_mm']:.2f} mm")
    if results["binary_contact_accuracy"] is not None:
        print(f"Binary contact accuracy: {results['binary_contact_accuracy'] * 100:.1f}%")
    if results["wrist_deg"] is not None:
        print(f"Wrist error: {results['wrist_deg']:.2f} deg")
        print(f"Final wrist error: {results['final_wrist_deg']:.2f} deg")
        if results["contact_wrist_deg"] is not None:
            print(f"Contact wrist error: {results['contact_wrist_deg']:.2f} deg")
        print(f"Wrist jitter: {results['wrist_jitter_deg']:.2f} deg")

    out = args.out or Path(f"results/eval_lstm_{args.hand}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"B3_SDF_LSTM": results}, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
