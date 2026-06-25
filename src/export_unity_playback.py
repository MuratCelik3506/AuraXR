"""Bir test segmentini Unity playback JSON'una aktar + ONNX'i Unity projesine kopyala.

Unity parite testi: Unity Inference Engine c2h_step.onnx'i frame-frame kosturup
'pca_expected' (torch free-running) ile karsilastirir.
"""
import os
import json
import shutil

import numpy as np
import torch

from dataset import GraspSegments
from model import ControllerToHand

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITY = "/Users/muratcelik/Desktop/Thesis/Unity/AURAXR/Assets"
ONNX_SRC = os.path.join(_REPO, "onnx", "c2h_step.onnx")


def main():
    ck = torch.load(os.path.join(_REPO, "checkpoints", "c2h_noprev.pt"),
                    map_location="cpu", weights_only=False)
    mean, std = ck["mean"], ck["std"]
    m = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"], use_prev=False)
    m.load_state_dict(ck["model"]); m.eval()

    ds = GraspSegments("test", mean, std)
    seg = next(s for s in ds.segments if s["obj_name"] == "mug_white")
    feat = seg["feat"].astype(np.float32); cat = seg["cat"].astype(np.int64)
    with torch.no_grad():
        pca_exp = m.forward_free(torch.as_tensor(feat)[None], torch.as_tensor(cat)[None]).numpy()[0]

    frames = []
    for t in range(len(feat)):
        frames.append(dict(
            feat=[float(x) for x in feat[t]],
            category=int(cat[t]),
            pca_expected=[float(x) for x in pca_exp[t]],
            pca_gt=[float(x) for x in seg["target"][t]],
            wrist_t=[float(x) for x in seg["wrist_t"][t]],
            wrist_q=[float(x) for x in seg["wrist_q"][t]],
            obj_t=[float(x) for x in seg["obj_t"][t]],
            obj_q=[float(x) for x in seg["obj_q"][t]],
            betas=[float(x) for x in seg["betas"][t]],
        ))
    out = dict(seq_id=seg["seq_id"], segment_id=seg["segment_id"], object=seg["obj_name"],
               hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
               input_mean=[float(x) for x in mean], input_std=[float(x) for x in std],
               n_frames=len(frames), frames=frames)

    sa = os.path.join(UNITY, "StreamingAssets")
    os.makedirs(sa, exist_ok=True)
    pj = os.path.join(sa, "c2h_playback.json")
    json.dump(out, open(pj, "w"))
    print("playback ->", pj, f"({len(frames)} frame, obj={seg['obj_name']})")

    # ONNX'i Unity'ye kopyala (import edilsin)
    md = os.path.join(UNITY, "AuraXR", "Models")
    os.makedirs(md, exist_ok=True)
    dst = os.path.join(md, "c2h_step.onnx")
    shutil.copy(ONNX_SRC, dst)
    print("onnx ->", dst)


if __name__ == "__main__":
    main()
