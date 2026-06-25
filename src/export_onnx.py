"""no_prev modelini tek-adim (stateful) ONNX'e export + onnxruntime parite (Faz 3 on adimi).

Runtime: Unity her frame -> normalize feat(13) + category_id + (h,c) -> ONNX -> pca15 + (h,c).
Normalizasyon ONNX disinda (Unity stats.json mean/std ile yapar).
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

from model import ControllerToHand, FEAT_DIM, POSE_DIM
from dataset import GraspSegments

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(_REPO, "onnx")


class StepModule(nn.Module):
    """Tek zaman-adimi: feat(B,1,13) + cat(B,1) + h,c -> pca(B,1,15) + h,c."""
    def __init__(self, m: ControllerToHand):
        super().__init__()
        self.cat_emb = m.cat_emb
        self.lstm = m.lstm
        self.head = m.head

    def forward(self, feat, cat, h0, c0):
        x = torch.cat([feat, self.cat_emb(cat)], dim=-1)   # no_prev
        out, (h, c) = self.lstm(x, (h0, c0))
        return self.head(out), h, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_REPO, "checkpoints", "c2h_noprev.pt"))
    args = ap.parse_args()
    os.makedirs(ONNX_DIR, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    assert ck["args"].get("no_prev", False), "Bu export no_prev model icin."
    hidden = ck["args"]["hidden"]; layers = ck["args"]["layers"]
    m = ControllerToHand(hidden=hidden, layers=layers, use_prev=False)
    m.load_state_dict(ck["model"]); m.eval()
    step = StepModule(m).eval()

    feat = torch.zeros(1, 1, FEAT_DIM)
    cat = torch.zeros(1, 1, dtype=torch.long)
    h0 = torch.zeros(layers, 1, hidden); c0 = torch.zeros(layers, 1, hidden)
    out_path = os.path.join(ONNX_DIR, "c2h_step.onnx")
    # legacy exporter (dynamo=False) -> agirliklar .onnx icine gomulu (harici .data yok),
    # Unity Inference Engine importer ile uyumlu. Sabit batch=1 (per-frame runtime).
    for stale in [out_path, out_path + ".data"]:
        if os.path.exists(stale):
            os.remove(stale)
    torch.onnx.export(
        step, (feat, cat, h0, c0), out_path,
        input_names=["feat", "category", "h0", "c0"],
        output_names=["pca15", "hn", "cn"],
        opset_version=17, dynamo=False,
    )
    print("export ->", out_path, f"({os.path.getsize(out_path)} bytes)")

    # stats + meta yaz (Unity icin)
    meta = dict(hidden=hidden, layers=layers, feat_dim=FEAT_DIM, pose_dim=POSE_DIM,
                input_mean=ck["mean"].tolist(), input_std=ck["std"].tolist(),
                feature_order=["rel_pos(3)", "rel_rot6d(6)", "rel_vel(3)", "dist(1)"],
                categories=["hook", "power", "wide"], use_prev=False)
    json.dump(meta, open(os.path.join(ONNX_DIR, "c2h_meta.json"), "w"), indent=2)

    # --- parite: torch free-running vs onnxruntime tek-adim ---
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ds = GraspSegments("test", ck["mean"], ck["std"])
    seg = next(s for s in ds.segments if s["obj_name"] == "mug_white")
    feat_np = seg["feat"].astype(np.float32); cat_np = seg["cat"].astype(np.int64)
    with torch.no_grad():
        torch_out = m.forward_free(torch.as_tensor(feat_np)[None],
                                   torch.as_tensor(cat_np)[None]).numpy()[0]
    h = np.zeros((layers, 1, hidden), np.float32); c = np.zeros((layers, 1, hidden), np.float32)
    onnx_out = np.empty_like(torch_out)
    for t in range(len(feat_np)):
        o, h, c = sess.run(None, {"feat": feat_np[t][None, None], "category": cat_np[t][None, None],
                                  "h0": h, "c0": c})
        onnx_out[t] = o[0, 0]
    md = float(np.abs(torch_out - onnx_out).max())
    print(f"parite max|torch-onnx| = {md:.2e}  ({'OK' if md < 1e-4 else 'FARK!'})")
    print("meta ->", os.path.join(ONNX_DIR, "c2h_meta.json"))


if __name__ == "__main__":
    main()
