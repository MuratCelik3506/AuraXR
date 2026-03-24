"""
Sliding Window Evaluation — Intent-Aware XR Framework
======================================================
Simulates real-time deployment by sliding a temporal window over 
a test sequence frame-by-frame.

Workflow:
---------
1. Load a full sequence (e.g., from H2O test set).
2. Buffer: Skip the first 10 seconds of data.
3. Simulation: For 20 seconds (600 frames at 30fps):
   - Take the last 30 frames as input.
   - Run IntentFormer inference.
   - Shift window by 1 frame (33ms).
4. Metrics: Compute Frame-wise Accuracy, Stability, and Convergence Point.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.h2o_dataset import load_sequence, wrist_relative_normalize, NUM_CLASSES
from src.models.intent_former import IntentFormer
from src.evaluate import (
    compute_stability, 
    find_convergence_point, 
    compute_frame_wise_accuracy,
    compute_mpjpe
)

# ─────────────────────────────────────────────────────────
# Simulation Logic
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_sliding_window(
    model: IntentFormer,
    seq_data: dict,
    device: torch.device,
    window_size: int = 30,
    buffer_frames: int = 0,
    sim_frames: int = 9999,
    obs_limit: float = 1.0, # Limit simulation to first X% of sequence
    target_pose: np.ndarray = None, # (2, 21, 3) target pose to compare against
) -> dict:
    model.eval()
    
    # Sequence info
    full_hand = seq_data["hand_poses"]      # (F, 2, 21, 3) - already wrist-relative from load_sequence
    full_obj  = seq_data["obj_poses_rt"]    # (F, 16)
    full_labels = seq_data["action_labels"] # (F,)
    full_raw_hand = seq_data["raw_hand_poses"]
    num_frames = seq_data["num_frames"]
    
    start_frame = buffer_frames
    
    # If sequence is shorter than buffer, start from beginning
    if start_frame >= num_frames:
        print(f"Warning: Sequence too short ({num_frames} frames) for {buffer_frames} frames buffer. Starting from frame 0.")
        start_frame = 0
        
    end_frame   = start_frame + sim_frames
    
    # Apply observation limit if requested
    limit_frame = int(num_frames * obs_limit)
    if end_frame > limit_frame:
        end_frame = limit_frame

    if end_frame > num_frames:
        end_frame = num_frames
        
    all_preds = []
    all_gt    = []
    all_probs = []
    all_mpjpe_next = [] # Error relative to NEXT frame
    all_dtt        = [] # Distance To Target (MPJPE relative to action end)
    all_pred_poses = []
    all_gt_poses   = [] 
    all_gt_obj_poses = []
    
    # Target pose is the final state of the action
    target_pose = full_hand[-1]

    print(f"Running simulation from Frame {start_frame} to Frame {end_frame}...")
    
    for t in tqdm(range(start_frame, end_frame)):
        # 1. Extract window [t-window_size, t]
        w_start = max(0, t - window_size)
        w_end   = t
        
        hand_window = full_hand[w_start:w_end]
        obj_window  = full_obj[w_start:w_end]
        
        # Padding if necessary
        curr_len = hand_window.shape[0]
        if curr_len < window_size:
            pad = window_size - curr_len
            hand_window = np.pad(hand_window, ((pad, 0), (0,0), (0,0), (0,0)))
            obj_window  = np.pad(obj_window,  ((pad, 0), (0,0)))
            
        # 2. Prepare tensors
        hand_flat = torch.from_numpy(hand_window.reshape(window_size, -1)).unsqueeze(0).to(device)
        obj_rt    = torch.from_numpy(obj_window).unsqueeze(0).to(device)
        obs_ratio = torch.tensor([min(1.0, t / num_frames)], dtype=torch.float32).to(device)
        
        # 3. Multi-Head Inference
        logits, pred_pose_tensor = model(hand_flat, obj_rt, obs_ratio)
        probs  = torch.softmax(logits, dim=-1)
        pred   = torch.argmax(probs, dim=-1).item()
        
        all_preds.append(int(pred))
        all_gt.append(int(full_labels[t] - 1)) 
        all_probs.append(float(probs.max().item()))
        
        # 4. Pose Accuracy Evaluation
        # A. Next-Frame Prediction Error (Regression)
        if t + 1 < num_frames:
            gt_next_pose   = full_hand[t+1]
            gt_next_pose_raw = full_raw_hand[t+1]
            wrist_pos = gt_next_pose_raw[:, 0:1, :] # (2, 1, 3)
            pred_next_pose = pred_pose_tensor.view(2, 21, 3).squeeze().cpu().numpy()
            
            pred_next_pose_camera = pred_next_pose + wrist_pos
            
            err_next = compute_mpjpe(pred_next_pose, gt_next_pose)
            all_mpjpe_next.append(err_next)
            all_pred_poses.append(pred_next_pose_camera.tolist())
            all_gt_poses.append(gt_next_pose_raw.tolist())
        else:
            all_mpjpe_next.append(0.0)
            all_pred_poses.append(full_raw_hand[t].tolist())
            all_gt_poses.append(full_raw_hand[t].tolist())

        # Collect GT Object Pose for 3D view
        all_gt_obj_poses.append(full_obj[t].tolist())

        # B. Distance to Target (DTT)
        dtt = compute_mpjpe(full_hand[t], target_pose)
        all_dtt.append(dtt)

    # Calculate metrics
    accuracy  = compute_frame_wise_accuracy(all_preds, all_gt)
    stability = compute_stability(all_preds)
    conv_pt   = find_convergence_point(all_preds, all_gt)
    
    return {
        "frame_wise_accuracy": accuracy,
        "stability":           stability,
        "convergence_point":   conv_pt,
        "predictions":         all_preds,
        "ground_truth":        all_gt,
        "max_probs":           all_probs,
        "pose_errors":         all_mpjpe_next, # MPJPE to next frame
        "dtt":                 all_dtt,        # Distance to target
        "predicted_poses":     all_pred_poses, 
        "gt_poses":            all_gt_poses,   # Added for comparison
        "gt_obj_poses":        all_gt_obj_poses,
        "mean_pose_error":     float(np.mean(all_mpjpe_next))
    }


# ─────────────────────────────────────────────────────────
# Main script
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model weights (.pt)")
    parser.add_argument("--data_root",  type=str, default="data/h2o")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of test sequences to evaluate")
    parser.add_argument("--use_synthetic", action="store_true", help="Generate synthetic data if missing")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model (MATCHING OPTIMIZED HYPERPARAMETERS)
    model = IntentFormer(
        d_model=256, 
        num_layers=6, 
        nhead=8, 
        dim_feedforward=1024,
        num_classes=NUM_CLASSES
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Warning: Running with randomly initialized model!")
        
    # Data source
    root = Path(args.data_root)
    if args.use_synthetic or not root.exists():
        print("Generating/using synthetic data for demonstration...")
        from src.tests.generate_synthetic_h2o import main as gen_main
        import sys
        old_argv = sys.argv
        sys.argv = ["generate_synthetic_h2o", "--out_dir", str(root), "--num_seqs", "10"]
        gen_main()
        sys.argv = old_argv
        
    # Pick sequences from test split
    test_split_file = root / "models" / "label_split" / "action_test.txt"
    with open(test_split_file) as f:
        lines = f.readlines()[1:] # skip header
        
    paths = sorted(list(set([line.strip().split()[1] for line in lines])))
    selected_paths = paths[:args.num_samples]
    
    results = []
    
    for rel_path in selected_paths:
        print(f"\nEvaluating sequence: {rel_path}")
        seq_dir = str(root / "annotations" / rel_path)
        seq_data = load_sequence(seq_dir)
        
        if seq_data is None:
            continue
            
        res = run_sliding_window(model, seq_data, device)
        results.append(res)
        
        print(f"  → Accuracy:  {res['frame_wise_accuracy']:.2%}")
        print(f"  → Stability: {res['stability']:.2%}")
        print(f"  → Conv Frame: {res['convergence_point']}")
        
        # Small visual timeline (first 100 frames of prediction)
        timeline_pred = "".join([str(p % 10) for p in res['predictions'][:100]])
        timeline_gt   = "".join([str(g % 10) for g in res['ground_truth'][:100]])
        print(f"  GT  : {timeline_gt}...")
        print(f"  PRED: {timeline_pred}...")

    if results:
        avg_acc = np.mean([r['frame_wise_accuracy'] for r in results])
        avg_stab = np.mean([r['stability'] for r in results])
        avg_conv = np.mean([r['convergence_point'] for r in results])
        
        print(f"\n{'═'*40}")
        print(f" OVERALL SLIDING WINDOW METRICS ({len(results)} samples)")
        print(f"{'═'*40}")
        print(f" Average Accuracy:  {avg_acc:.2%}")
        print(f" Average Stability: {avg_stab:.2%}")
        print(f" Average Conv Frame: {avg_conv:.1f}")
        print(f"{'═'*40}")

if __name__ == "__main__":
    main()
