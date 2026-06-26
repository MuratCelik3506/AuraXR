"""Tum varyantlari TEST'te ayni harness ile karsilastir (eval_metrics tek dogruluk kaynagi).

Sutunlar: MPJPE, parmak-ucu, cat_mean'e karsi kazanc, jerk, kategori kirilimi.
Ana 5 varyant + 3 mimari + statik cat_mean baseline.
"""
import os
import json
import torch

from dataset import load_stats
from model import ControllerToHand
from mano_right import ManoRight
from eval_metrics import full_eval, device

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(_REPO, "checkpoints")
RESULTS = os.path.join(_REPO, "results")

VARIANTS = [
    ("baseline(noprev)", "c2h_noprev.pt"),
    ("A: fk_loss", "c2h_fk.pt"),
    ("C: obj_size", "c2h_objsize.pt"),
    ("A+C: fk+obj_size", "c2h_fk_objsize.pt"),
    ("B: AR+sched_samp", "c2h_ss.pt"),
    ("arch: LSTM-ff", "c2h_arch_lstm.pt"),
    ("arch: TCN", "c2h_arch_tcn.pt"),
    ("arch: Transformer", "c2h_arch_transformer.pt"),
]


def eval_ckpt(path, mano, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    obj_size = ck["args"].get("obj_size", False)
    feat_dim = 16 if obj_size else 13
    model = ControllerToHand(hidden=ck["args"]["hidden"], layers=ck["args"]["layers"],
                             use_prev=not ck["args"].get("no_prev", False),
                             feat_dim=feat_dim, arch=ck["args"].get("arch", "lstm")).to(dev)
    model.load_state_dict(ck["model"])
    tag = os.path.basename(path)[4:-3]   # c2h_<tag>.pt
    return full_eval(model, mano, ck["mean"], ck["std"], obj_size, dev, "test", tag,
                     write=False, cap=8000)


def main():
    dev = device()
    mano = ManoRight()
    table = {}
    static_mm = None
    for name, fn in VARIANTS:
        p = os.path.join(CKPT, fn)
        if not os.path.exists(p):
            print("ATLA (yok):", fn); continue
        r = eval_ckpt(p, mano, dev)
        table[name] = r
        static_mm = r["cat_mean_baseline_mm"]
    if static_mm is not None:
        table["baseline: cat_mean"] = dict(mpjpe_mm=static_mm, fingertip_mm=None,
                                           gain_vs_static_mm=0.0, jerk_free=None, by_category={})

    json.dump(table, open(os.path.join(RESULTS, "eval_compare.json"), "w"), indent=2)
    print(f"\n=== TEST free-running (mm, lower=better) ===")
    print(f"  {'varyant':22s} {'MPJPE':>6s} {'tip':>6s} {'kazanc':>7s} {'jerk':>6s}   per-cat")
    for name, r in sorted(table.items(), key=lambda kv: kv[1]["mpjpe_mm"]):
        pc = " ".join(f"{k}={v['mpjpe']}" for k, v in r.get("by_category", {}).items())
        tip = f"{r['fingertip_mm']:.1f}" if r.get("fingertip_mm") is not None else "  -"
        jk = f"{r['jerk_free']:.3f}" if r.get("jerk_free") is not None else "  -"
        gn = f"{r.get('gain_vs_static_mm', 0):+.2f}"
        print(f"  {name:22s} {r['mpjpe_mm']:6.2f} {tip:>6s} {gn:>7s} {jk:>6s}   {pc}")
    print("\nrapor -> results/eval_compare.json")


if __name__ == "__main__":
    main()
