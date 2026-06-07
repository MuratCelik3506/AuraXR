"""export_onnx.py — Step 6: export trained AuraXRModel to ONNX for Unity Sentis.

Run:
    python export_onnx.py --checkpoint ../checkpoints/right/ --output_dir ../onnx/
    python export_onnx.py --checkpoint ../checkpoints/left/  --output_dir ../onnx/

ONNX input/output spec:
    Input  "spatial_input": shape [1, 8]  — [dir_world(3), dir_obj_local(3), dist(1), approach_speed(1)], normalized
    Input  "object_input":  shape [1, 7]  — [grip_onehot(4), bbox(3)], normalized
    Output "joint_angles":  shape [1, 22] — normalized; denorm with target_mean/std from model_meta.json
    Output "wrist_rot_6d":  shape [1,  6] — normalized; denorm with wrist_rot_mean/std from model_meta.json
                                            Then Gram-Schmidt decode → rotation matrix → Quaternion in Unity

Unity denormalization:
    float angle_i  = (joint_angles[i]  * target_std[i])   + target_mean[i];
    float rot6d_i  = (wrist_rot_6d[i]  * wrist_rot_std[i]) + wrist_rot_mean[i];
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from model import AuraXRModel

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def parse_args():
    p = argparse.ArgumentParser(description="Export AuraXR model to ONNX.")
    p.add_argument("--checkpoint",  required=True, type=Path,
                   help="Directory with best_model.pt and model_meta.json.")
    p.add_argument("--output_dir",  default=Path("onnx"), type=Path)
    p.add_argument("--opset",       default=14, type=int,
                   help="ONNX opset version (Unity Sentis supports 14+).")
    return p.parse_args()


def main():
    args = parse_args()

    meta_path  = args.checkpoint / "model_meta.json"
    model_path = args.checkpoint / "best_model.pt"

    if not model_path.exists():
        print(f"[ERROR] {model_path} not found. Run train.py first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    arch = meta["architecture"]
    model = AuraXRModel(
        spatial_input_dim=arch["spatial_input_dim"],
        object_input_dim=arch["object_input_dim"],
        hidden_dim=arch["hidden_dim"],
        embedding_dim=arch["embedding_dim"],
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Dummy inputs with batch size 1
    spatial_dummy = torch.zeros(1, arch["spatial_input_dim"])
    object_dummy  = torch.zeros(1, arch["object_input_dim"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Infer hand side from checkpoint dir name
    hand = args.checkpoint.name  # "right" or "left"
    onnx_path = args.output_dir / f"auraxr_{hand}.onnx"
    meta_out   = args.output_dir / f"model_meta_{hand}.json"

    torch.onnx.export(
        model,
        (spatial_dummy, object_dummy),
        str(onnx_path),
        input_names=["spatial_input", "object_input"],
        output_names=["joint_angles", "wrist_rot_6d"],
        dynamic_axes={
            "spatial_input": {0: "batch_size"},
            "object_input":  {0: "batch_size"},
            "joint_angles":  {0: "batch_size"},
            "wrist_rot_6d":  {0: "batch_size"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported: {onnx_path}")

    # Verify with ONNX Runtime
    if HAS_ORT:
        sess = ort.InferenceSession(str(onnx_path))
        inputs = {
            "spatial_input": spatial_dummy.numpy(),
            "object_input":  object_dummy.numpy(),
        }
        outputs = sess.run(None, inputs)
        joint_shape, rot_shape = outputs[0].shape, outputs[1].shape
        print(f"ONNX Runtime verification: joint_angles={joint_shape}  wrist_rot_6d={rot_shape}")
        assert joint_shape == (1, arch["output_dim"]), f"Unexpected joint shape: {joint_shape}"
        assert rot_shape   == (1, 6),                  f"Unexpected rot shape: {rot_shape}"
        print("  Shapes OK.")

        # Cross-check against PyTorch
        with torch.no_grad():
            pt_joints, pt_rot = model(spatial_dummy, object_dummy)
        max_diff_joints = abs(outputs[0] - pt_joints.numpy()).max()
        max_diff_rot    = abs(outputs[1] - pt_rot.numpy()).max()
        print(f"  Max abs diff (joints): {max_diff_joints:.2e}  (rot): {max_diff_rot:.2e}")
        assert max_diff_joints < 1e-5, f"Large joint diff: {max_diff_joints}"
        assert max_diff_rot    < 1e-5, f"Large rot diff: {max_diff_rot}"
        print("  Numerical match OK.")
    else:
        print("  onnxruntime not installed — skipping verification.")

    # Copy model_meta.json for Unity
    shutil.copy(meta_path, meta_out)
    print(f"Copied: {meta_out}")

    print(f"\nReady for Unity:")
    print(f"  Copy {onnx_path} → Unity Assets/AuraXR/Models/")
    print(f"  Copy {meta_out}  → Unity Assets/AuraXR/Models/")
    print(f"\nUnity denormalization (C#):")
    print(f"  float angle_i  = (joint_angles[i]  * target_std[i])    + target_mean[i];")
    print(f"  float rot6d_i  = (wrist_rot_6d[i]  * wrist_rot_std[i]) + wrist_rot_mean[i];")
    print(f"  Then Gram-Schmidt decode rot6d → Quaternion in AuraXRInferenceManager.cs")


if __name__ == "__main__":
    main()
