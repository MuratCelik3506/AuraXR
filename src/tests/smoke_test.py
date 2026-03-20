"""
Smoke Test — Full Pipeline
===========================
Tests the entire pipeline end-to-end using ONLY synthetic data.
No real H2O dataset required.

Checks:
  1. Data loader & parser           (h2o_dataset.py)
  2. Model forward pass             (intent_former.py)
  3. EarlyPredictionLoss            (intent_former.py)
  4. Training step                  (train.py helpers)
  5. Evaluation metrics             (evaluate.py)
  6. DataLoader batch iteration     (h2o_dataset.py)

Usage:
    python -m src.tests.smoke_test
    python -m src.tests.smoke_test --data_root data/h2o  # reuse existing synthetic data
    python -m src.tests.smoke_test --keep_data           # don't delete generated data after

Exit code 0 = all tests passed.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


# ── coloured output helpers ─────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

PASS   = f"{GREEN}  ✓ PASS{RESET}"
FAIL   = f"{RED}  ✗ FAIL{RESET}"


def run_test(name: str, fn):
    print(f"\n{'─'*60}")
    print(f"TEST: {name}")
    try:
        fn()
        print(PASS)
        return True
    except Exception as exc:
        print(f"{FAIL}  →  {exc}")
        import traceback
        traceback.print_exc()
        return False


# ── individual tests ─────────────────────────────────────────────────────────

def test_synthetic_generation(data_root: str):
    """Ensure synthetic data generator produces expected file structure."""
    from src.tests.generate_synthetic_h2o import main as gen_main
    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ["generate_synthetic_h2o", "--out_dir", data_root, "--num_seqs", "6"]
    gen_main()
    _sys.argv = old_argv

    root = Path(data_root)
    assert (root / "models" / "label_split" / "action_train.txt").exists(), \
        "action_train.txt missing"
    assert (root / "models" / "label_split" / "action_val.txt").exists(), \
        "action_val.txt missing"
    assert (root / "models" / "label_split" / "action_test.txt").exists(), \
        "action_test.txt missing"

    anno_dirs = list((root / "annotations" / "subject1" / "h1").iterdir())
    assert len(anno_dirs) > 0, "No annotation sequence directories created"


def test_parsers(data_root: str):
    """Test low-level file parsers."""
    from src.data.h2o_dataset import (
        _parse_hand_pose, _parse_obj_pose_rt, _parse_action_label,
        NUM_HANDS, NUM_JOINTS, JOINT_DIM
    )
    root = Path(data_root)
    seq_dirs = sorted((root / "annotations" / "subject1" / "h1").iterdir())
    cam_dir  = seq_dirs[0] / "cam4"

    hp_file  = sorted((cam_dir / "hand_pose").glob("*.txt"))[0]
    op_file  = sorted((cam_dir / "obj_pose_rt").glob("*.txt"))[0]
    al_file  = sorted((cam_dir / "action_label").glob("*.txt"))[0]

    hp = _parse_hand_pose(str(hp_file))
    assert hp.shape == (NUM_HANDS, NUM_JOINTS, JOINT_DIM), \
        f"Expected (2,21,3), got {hp.shape}"

    op = _parse_obj_pose_rt(str(op_file))
    assert op.shape == (16,), f"Expected (16,), got {op.shape}"

    al = _parse_action_label(str(al_file))
    assert isinstance(al, int), "action label should be int"
    assert 1 <= al <= 36, f"label {al} out of range [1,36]"


def test_load_sequence(data_root: str):
    """Test sequence loading + wrist-relative normalization."""
    from src.data.h2o_dataset import load_sequence, NUM_HANDS, NUM_JOINTS, JOINT_DIM
    root = Path(data_root)
    seq_dirs = sorted((root / "annotations" / "subject1" / "h1").iterdir())
    seq_dir  = str(seq_dirs[0])

    seq = load_sequence(seq_dir)
    assert seq is not None, "load_sequence returned None"

    hp = seq["hand_poses"]       # (F, 2, 21, 3)
    assert hp.shape[1:] == (NUM_HANDS, NUM_JOINTS, JOINT_DIM), \
        f"Unexpected hand_poses shape: {hp.shape}"

    # Wrist should be all zeros after normalization
    wrists = hp[:, :, 0, :]   # (F, 2, 3)
    assert np.allclose(wrists, 0, atol=1e-5), \
        "Wrist-relative normalization failed: wrist != 0"


def test_extract_window(data_root: str):
    """Test window extraction at various obs_ratios."""
    from src.data.h2o_dataset import load_sequence, extract_window
    root = Path(data_root)
    seq_dirs = sorted((root / "annotations" / "subject1" / "h1").iterdir())
    seq = load_sequence(str(seq_dirs[0]))

    for obs_ratio in [0.20, 0.25, 0.30]:
        w = extract_window(seq, start_act=20, end_act=80,
                           obs_ratio=obs_ratio, window_size=30)
        assert w is not None, f"extract_window returned None at ratio={obs_ratio}"
        assert w["hand_flat"].shape == (30, 126), \
            f"hand_flat shape wrong: {w['hand_flat'].shape}"
        assert w["obj_rt"].shape == (30, 16), \
            f"obj_rt shape wrong: {w['obj_rt'].shape}"


def test_dataset_and_loader(data_root: str):
    """Test H2ODataset __len__ and __getitem__, then DataLoader batching."""
    from src.data.h2o_dataset import H2ODataset
    from torch.utils.data import DataLoader

    ds = H2ODataset(data_root, split="train", window_size=30)
    assert len(ds) > 0, "Dataset has 0 samples"

    item = ds[0]
    assert item["hand_flat"].shape == (30, 126), \
        f"hand_flat: {item['hand_flat'].shape}"
    assert item["obj_rt"].shape == (30, 16), \
        f"obj_rt: {item['obj_rt'].shape}"
    assert item["label"].dtype == torch.long, "label should be int64"
    assert item["obs_ratio"].ndim == 0, "obs_ratio should be scalar"

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["hand_flat"].shape[0] <= 4
    assert batch["hand_flat"].shape[1:] == (30, 126)


def test_model_forward():
    """Test model instantiation and forward pass shapes."""
    from src.models.intent_former import IntentFormer

    model = IntentFormer()
    model.eval()
    B, T = 4, 30
    hand  = torch.randn(B, T, 126)
    obj   = torch.randn(B, T, 16)
    obs   = torch.rand(B)

    with torch.no_grad():
        logits = model(hand, obj, obs)
    assert logits.shape == (B, 36), \
        f"Expected logits (4, 36), got {logits.shape}"


def test_model_predict_proba():
    """Test predict_proba sums to 1."""
    from src.models.intent_former import IntentFormer

    model = IntentFormer()
    model.eval()
    B, T = 2, 30
    proba = model.predict_proba(
        torch.randn(B, T, 126),
        torch.randn(B, T, 16),
        torch.rand(B),
    )
    assert proba.shape == (B, 36)
    sums = proba.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
        "predict_proba does not sum to 1"


def test_early_prediction_loss():
    """Test EarlyPredictionLoss is a valid scalar > 0 and is differentiable."""
    from src.models.intent_former import EarlyPredictionLoss

    criterion = EarlyPredictionLoss(alpha=2.0)
    B, C = 8, 36

    # Use a tiny trainable Linear so logits have a grad_fn
    linear    = torch.nn.Linear(16, C)
    x         = torch.randn(B, 16)
    logits    = linear(x)                           # has grad_fn through Linear
    labels    = torch.randint(0, C, (B,))
    obs_ratio = torch.rand(B)

    loss = criterion(logits, labels, obs_ratio)
    assert loss.ndim == 0,   "Loss should be a scalar"
    assert loss.item() > 0,  "Loss should be positive"
    loss.backward()          # must be differentiable (grads flow to linear.weight)
    assert linear.weight.grad is not None, "No gradient flowed to Linear.weight"


def test_one_training_step(data_root: str):
    """Run a single optimizer step end-to-end."""
    from src.data.h2o_dataset import H2ODataset, NUM_CLASSES
    from src.models.intent_former import IntentFormer, EarlyPredictionLoss
    from torch.utils.data import DataLoader
    import torch.optim as optim

    ds     = H2ODataset(data_root, split="train", window_size=30)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch  = next(iter(loader))

    device = torch.device("cpu")
    model  = IntentFormer(num_classes=NUM_CLASSES).to(device)
    crit   = EarlyPredictionLoss()
    opt    = optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    hand   = batch["hand_flat"].to(device)
    obj    = batch["obj_rt"].to(device)
    obs    = batch["obs_ratio"].to(device)
    labels = batch["label"].to(device)

    opt.zero_grad()
    logits = model(hand, obj, obs)
    loss   = crit(logits, labels, obs)
    loss.backward()
    opt.step()

    assert loss.item() > 0, "Training step produced zero or negative loss"


def test_compute_metrics():
    """Test compute_metrics with known values."""
    from src.evaluate import compute_metrics

    # Perfect predictions for the first 5 classes
    preds  = torch.tensor([0, 1, 2, 3, 4])
    labels = torch.tensor([0, 1, 2, 3, 4])
    m = compute_metrics(preds, labels, num_classes=36)

    assert m["accuracy"] == 1.0, f"Expected 1.0 accuracy, got {m['accuracy']}"
    assert "precision" in m
    assert "recall"    in m
    assert "f1"        in m

    # Completely wrong predictions
    preds_wrong  = torch.tensor([1, 2, 3, 4, 5])
    m2 = compute_metrics(preds_wrong, labels, num_classes=36)
    assert m2["accuracy"] == 0.0, f"Expected 0.0 accuracy, got {m2['accuracy']}"


def test_precision_at_obs_ratio():
    """Test precision_at_obs_ratio bucketing."""
    from src.evaluate import precision_at_obs_ratio

    # All correct at obs_ratio=0.20
    preds      = [0, 1, 2]
    labels     = [0, 1, 2]
    obs_ratios = [0.20, 0.20, 0.20]
    result = precision_at_obs_ratio(preds, labels, obs_ratios)
    assert result["≤0.20"] == 1.0, f"Expected 1.0, got {result['≤0.20']}"


def test_compute_ttc():
    """Test TTC formula."""
    from src.evaluate import compute_ttc

    # action 20-80 (61 frames), obs_ratio=0.20 → obs_end = 20 + int(61*0.20) = 32
    # TTC = (80 - 32) / 30 ≈ 1.60 s
    ttc = compute_ttc(start_act=20, end_act=80, obs_ratio=0.20, fps=30)
    assert abs(ttc - 1.6) < 0.01, f"Expected ≈1.60, got {ttc}"


def test_full_evaluate_model(data_root: str):
    """Run evaluate_model on a random model to confirm output dict keys."""
    from src.data.h2o_dataset import H2ODataset
    from src.models.intent_former import IntentFormer
    from src.evaluate import evaluate_model
    from torch.utils.data import DataLoader

    ds     = H2ODataset(data_root, split="val", window_size=30)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    model  = IntentFormer()
    device = torch.device("cpu")

    result = evaluate_model(model, loader, device)
    required_keys = ["accuracy", "precision", "recall", "f1",
                     "ghosting_trigger_rate"]
    for k in required_keys:
        assert k in result, f"Missing key '{k}' in evaluate_model output"


# ── runner ───────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Smoke-test the full XR pipeline")
    ap.add_argument("--data_root",   default="",
                    help="Reuse an existing synthetic dataset. "
                         "If empty, a temp dir is created automatically.")
    ap.add_argument("--keep_data",   action="store_true",
                    help="Don't delete the generated synthetic data after tests")
    return ap.parse_args()


def main():
    args = parse_args()

    # Decide on the data directory
    tmp_dir     = None
    own_tmpdir  = False
    if args.data_root:
        data_root = args.data_root
    else:
        tmp_dir   = tempfile.mkdtemp(prefix="xr_smoke_")
        data_root = tmp_dir
        own_tmpdir = True

    print(f"\n{'═'*60}")
    print(f"  XR Intent Framework — Smoke Test Suite")
    print(f"  data_root : {data_root}")
    print(f"{'═'*60}")

    results = {}

    # (1) Data generation — must run first
    results["Synthetic data generation"] = run_test(
        "Synthetic data generation",
        lambda: test_synthetic_generation(data_root)
    )

    if not results["Synthetic data generation"]:
        print(f"\n{RED}Cannot continue: data generation failed.{RESET}")
        sys.exit(1)

    # (2) Low-level parsers
    results["File parsers (_parse_hand_pose etc.)"] = run_test(
        "File parsers (_parse_hand_pose etc.)",
        lambda: test_parsers(data_root)
    )

    # (3) Sequence loading + wrist-relative normalization
    results["load_sequence + wrist normalization"] = run_test(
        "load_sequence + wrist normalization",
        lambda: test_load_sequence(data_root)
    )

    # (4) Window extraction
    results["extract_window (obs_ratios 0.20/0.25/0.30)"] = run_test(
        "extract_window (obs_ratios 0.20/0.25/0.30)",
        lambda: test_extract_window(data_root)
    )

    # (5) Dataset and DataLoader
    results["H2ODataset __getitem__ + DataLoader"] = run_test(
        "H2ODataset __getitem__ + DataLoader",
        lambda: test_dataset_and_loader(data_root)
    )

    # (6)-(8) Model
    results["IntentFormer forward pass"] = run_test(
        "IntentFormer forward pass",
        test_model_forward
    )
    results["IntentFormer predict_proba"] = run_test(
        "IntentFormer predict_proba",
        test_model_predict_proba
    )
    results["EarlyPredictionLoss backward"] = run_test(
        "EarlyPredictionLoss backward",
        test_early_prediction_loss
    )

    # (9) Integration: one training step with real data
    results["One training step (data → model → loss → backward)"] = run_test(
        "One training step (data → model → loss → backward)",
        lambda: test_one_training_step(data_root)
    )

    # (10)-(12) Evaluation utilities
    results["compute_metrics (precision/recall/f1/acc)"] = run_test(
        "compute_metrics (precision/recall/f1/acc)",
        test_compute_metrics
    )
    results["precision_at_obs_ratio"] = run_test(
        "precision_at_obs_ratio",
        test_precision_at_obs_ratio
    )
    results["compute_ttc formula"] = run_test(
        "compute_ttc formula",
        test_compute_ttc
    )

    # (13) Full evaluate_model loop
    results["evaluate_model full loop"] = run_test(
        "evaluate_model full loop",
        lambda: test_full_evaluate_model(data_root)
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    passed = sum(v for v in results.values())
    total  = len(results)
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        color = GREEN if ok else RED
        print(f"  {color}{icon}{RESET}  {name}")
    print(f"{'─'*60}")
    color = GREEN if passed == total else RED
    print(f"  {color}{passed}/{total} tests passed{RESET}")
    print(f"{'═'*60}\n")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if own_tmpdir and not args.keep_data and tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
