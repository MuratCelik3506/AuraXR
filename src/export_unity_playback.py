"""Test segmentlerini Unity playback JSON'larina aktar + ONNX'i Unity projesine kopyala.

Unity parite testi: Unity Inference Engine c2h_step.onnx'i frame-frame kosturup
'pca_expected' (torch free-running) ile karsilastirir.

Her senaryo icin ayrica MANO FK eklem dunya-konumlari da export edilir:
  - pred_joints: free-running tahminin (pca_expected) 16-eklem dunya konumu
  - gt_joints:   ground-truth (finger_aa45) 16-eklem dunya konumu
Unity tarafi bunlari basit bir iskelet (kure + kemik) olarak cizip yan yana
karsilastirma yapar; tum eklemler HOT3D dunya frame'inde, wrist/obj ile ayni.
"""
import os
import json
import shutil

import numpy as np
import torch

from dataset import GraspSegments
from model import ControllerToHand
from mano_torch import ManoTorchFK

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITY = "/Users/muratcelik/Desktop/Thesis/Unity/AURAXR/Assets"
ONNX_SRC = os.path.join(_REPO, "onnx", "c2h_step.onnx")

# Yan yana test edilecek senaryolar (test setindeki obj_name'ler).
# Her biri c2h_playback_<obj>.json olarak yazilir; ilki ayrica varsayilan
# c2h_playback.json'a da yazilir (mevcut tek-driver ile geri-uyum).
SCENARIOS = ["mug_white", "mouse", "bowl", "can_soup", "mug_patterned"]


def _world_joints(fk, pose, seg):
    """pose (T,15 pca) veya (T,45 aa) -> (T,16,3) HOT3D dunya eklem konumu."""
    with torch.no_grad():
        j = fk.joints(
            torch.as_tensor(pose, dtype=torch.float32),
            torch.as_tensor(seg["betas"], dtype=torch.float32),
            torch.as_tensor(seg["wrist_q"], dtype=torch.float32),
            torch.as_tensor(seg["wrist_t"], dtype=torch.float32),
        )
    return j.numpy()


def export_one(ds, model, fk, mean, std, hidden, layers, obj_name):
    seg = next((s for s in ds.segments if s["obj_name"] == obj_name), None)
    if seg is None:
        print(f"  ! senaryo atlandi (test setinde yok): {obj_name}")
        return None

    feat = seg["feat"].astype(np.float32)
    cat = seg["cat"].astype(np.int64)
    with torch.no_grad():
        pca_exp = model.forward_free(
            torch.as_tensor(feat)[None], torch.as_tensor(cat)[None]
        ).numpy()[0]

    pred_joints = _world_joints(fk, pca_exp, seg)       # (T,16,3) free-running tahmin
    gt_joints = _world_joints(fk, seg["target"], seg)   # (T,16,3) ground-truth

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
            pred_joints=[float(x) for x in pred_joints[t].reshape(-1)],  # 16*3
            gt_joints=[float(x) for x in gt_joints[t].reshape(-1)],      # 16*3
        ))

    return dict(
        seq_id=seg["seq_id"], segment_id=seg["segment_id"], object=seg["obj_name"],
        scenario=obj_name,
        hidden=hidden, layers=layers,
        input_mean=[float(x) for x in mean], input_std=[float(x) for x in std],
        joint_parents=[int(p) for p in fk.parents],  # 16 (MANO kintree, [-1,0,1,2,...])
        n_frames=len(frames), frames=frames,
    )


def main():
    ck = torch.load(os.path.join(_REPO, "checkpoints", "c2h_noprev.pt"),
                    map_location="cpu", weights_only=False)
    mean, std = ck["mean"], ck["std"]
    hidden = ck["args"]["hidden"]
    layers = ck["args"]["layers"]
    m = ControllerToHand(hidden=hidden, layers=layers, use_prev=False)
    m.load_state_dict(ck["model"]); m.eval()

    fk = ManoTorchFK()
    ds = GraspSegments("test", mean, std)

    sa = os.path.join(UNITY, "StreamingAssets")
    os.makedirs(sa, exist_ok=True)

    written = []
    for i, obj in enumerate(SCENARIOS):
        out = export_one(ds, m, fk, mean, std, hidden, layers, obj)
        if out is None:
            continue
        pj = os.path.join(sa, f"c2h_playback_{obj}.json")
        json.dump(out, open(pj, "w"))
        written.append(os.path.basename(pj))
        print(f"playback -> {pj}  ({out['n_frames']} frame, obj={obj})")
        # Ilk senaryoyu ayrica varsayilan dosyaya da yaz (mevcut driver icin).
        if i == 0:
            dj = os.path.join(sa, "c2h_playback.json")
            json.dump(out, open(dj, "w"))
            print(f"playback -> {dj}  (varsayilan, {obj})")

    print(f"\n{len(written)} senaryo yazildi: {', '.join(written)}")

    # ONNX'i Unity'ye kopyala (import edilsin)
    md = os.path.join(UNITY, "AuraXR", "Models")
    os.makedirs(md, exist_ok=True)
    dst = os.path.join(md, "c2h_step.onnx")
    shutil.copy(ONNX_SRC, dst)
    print("onnx ->", dst)


if __name__ == "__main__":
    main()
