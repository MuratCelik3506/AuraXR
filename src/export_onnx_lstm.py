"""Export AuraXR SDF-LSTM checkpoints to ONNX for Unity."""

import argparse
import json
import sys
from pathlib import Path

import torch
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from model import SDFLSTMModel


DEFAULT_INPUT_DIM = 29
EMBED_DIM = 32
HIDDEN = 256
LAYERS = 2
POSE_DIM = 15


def infer_feat_dim(state_dict: dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("feat_proj.0.weight")
    return int(weight.shape[1]) if weight is not None else DEFAULT_INPUT_DIM


def infer_orientation_aware(state_dict: dict[str, torch.Tensor]) -> bool:
    weight = state_dict.get("obj_inj.0.weight")
    return bool(weight is not None and int(weight.shape[1]) == 99)


def export_lstm(ckpt_path: Path, out_dir: Path, hand: str, opset: int = 14) -> tuple[Path, int, bool]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_dim = infer_feat_dim(ckpt["model"])
    orientation_aware = infer_orientation_aware(ckpt["model"])
    model = SDFLSTMModel(feat_dim=input_dim, orientation_aware_sdf=orientation_aware)
    model.load_state_dict(ckpt["model"])
    model.eval()

    frame_feat = torch.zeros(1, input_dim)
    obj_embed = torch.zeros(1, EMBED_DIM)
    h_0 = torch.zeros(LAYERS, 1, HIDDEN)
    c_0 = torch.zeros(LAYERS, 1, HIDDEN)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"auraxr_lstm_{hand}.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model,
            (frame_feat, obj_embed, h_0, c_0),
            str(out_path),
            opset_version=opset,
            input_names=["frame_feat", "obj_embed", "h_0", "c_0"],
            output_names=["pose_pca", "wrist_rot", "contact_prob", "h_n", "c_n"],
            dynamic_axes={
                "frame_feat": {0: "batch"},
                "obj_embed": {0: "batch"},
                "h_0": {1: "batch"},
                "c_0": {1: "batch"},
                "pose_pca": {0: "batch"},
                "wrist_rot": {0: "batch"},
                "contact_prob": {0: "batch"},
                "h_n": {1: "batch"},
                "c_n": {1: "batch"},
            },
            dynamo=False,
        )

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    pose, wrist, contact, h_n, c_n = sess.run(None, {
        "frame_feat": frame_feat.numpy(),
        "obj_embed": obj_embed.numpy(),
        "h_0": h_0.numpy(),
        "c_0": c_0.numpy(),
    })
    assert pose.shape == (1, POSE_DIM)
    assert wrist.shape == (1, 6)
    assert contact.shape == (1, 1)
    assert h_n.shape == (LAYERS, 1, HIDDEN)
    assert c_n.shape == (LAYERS, 1, HIDDEN)
    print(f"Exported and validated {out_path}")
    return out_path, input_dim, orientation_aware


def export_meta(out_dir: Path, hand: str, data_dir: Path | None, input_dim: int, orientation_aware: bool):
    import h5py

    meta = {
        "architecture": {
            "model_type": "sdf_lstm",
            "input_dim": input_dim,
            "embed_dim": EMBED_DIM,
            "lstm_hidden": HIDDEN,
            "lstm_layers": LAYERS,
            "output_dim": POSE_DIM,
            "output_type": "mano_pca",
            "orientation_aware_sdf": orientation_aware,
        },
        "feature_mean": [0.0] * max(0, input_dim - 4),
        "feature_std": [1.0] * max(0, input_dim - 4),
        "sdf_mean": [0.0] * 4,
        "sdf_std": [1.0] * 4,
        "target_mean": [0.0] * 15,
        "target_std": [1.0] * 15,
        "wrist_rot_mean": [0.0] * 6,
        "wrist_rot_std": [1.0] * 6,
    }

    h5_path = data_dir / "dataset_mano.h5" if data_dir else None
    if h5_path and h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            stored = json.loads(f.attrs["meta"])
        for key in meta:
            if key != "architecture" and key in stored:
                meta[key] = stored[key]
        if "architecture" in stored:
            meta["architecture"].update(stored["architecture"])
            meta["architecture"]["model_type"] = "sdf_lstm"
            meta["architecture"]["input_dim"] = input_dim
    meta["architecture"]["orientation_aware_sdf"] = orientation_aware

    out = out_dir / f"model_meta_lstm_{hand}.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out_dir", default="onnx", type=Path)
    parser.add_argument("--hand", required=True, choices=["left", "right"])
    parser.add_argument("--data_dir", default=None, type=Path)
    args = parser.parse_args()

    _, input_dim, orientation_aware = export_lstm(args.ckpt, args.out_dir, args.hand)
    export_meta(args.out_dir, args.hand, args.data_dir, input_dim, orientation_aware)


if __name__ == "__main__":
    main()
