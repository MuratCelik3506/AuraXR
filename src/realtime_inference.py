"""
Real-Time Webcam Inference with MediaPipe + Ghost Hand Visualization
=====================================================================

This script enables live hand pose estimation using MediaPipe and 
real-time intent prediction with the IntentFormer model.

Features:
---------
1. Webcam capture + MediaPipe hand detection
2. Sliding window (21 frames) for temporal context
3. Real-time intent prediction at different observation ratios
4. Ghost hand visualization (predicted next frame)
5. FPS monitoring and latency measurement

Usage:
------
    python src/realtime_inference.py \
        --checkpoint checkpoints/combined_best/best_model.pt \
        --device mps \
        --fusion shared_head
"""

import os
import sys
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] MediaPipe not installed. Install with: pip install mediapipe")
    sys.exit(1)

from src.models.intent_former import IntentFormer

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─────────────────────────────────────────────────────────
# Intent Labels (shared_head mapping)
# ─────────────────────────────────────────────────────────

INTENT_LABELS_SHARED_HEAD = {
    0: "Pickup 🎯",      # Approach / Grasp
    1: "Manipulate ✋",   # Steady state
    2: "Release 📦",      # Put down
}

INTENT_LABELS_CONCAT = {i: f"Action {i}" for i in range(36)}

# Color palette for visualization
COLOR_CONFIDENT = (0, 255, 0)    # Green
COLOR_UNCERTAIN = (0, 165, 255)  # Orange
COLOR_LOW_CONFIDENCE = (0, 0, 255)  # Red
COLOR_GHOST = (200, 150, 255)    # Purple (ghost hand)


# ─────────────────────────────────────────────────────────
# Utility: Extract hand landmarks from MediaPipe
# ─────────────────────────────────────────────────────────

def extract_hand_landmarks(image_h, image_w, hands_results):
    """
    Extract hand landmarks from MediaPipe and return a wrist-relative
    126-dim vector for left and right hands.
    
    MediaPipe returns 21 landmarks per hand (x, y, z).
    Output:
        landmarks:    (126,) = 21 joints × 3 coords × 2 hands (wrist-relative)
        wrist_anchor: (2, 3) absolute wrist position per hand [left, right]
        hand_present: (2,) bool mask indicating which hands were detected
    """
    landmarks = np.zeros((126,), dtype=np.float32)
    wrist_anchor = np.zeros((2, 3), dtype=np.float32)
    hand_present = np.zeros((2,), dtype=bool)
    
    if hands_results.multi_hand_landmarks:
        for hand_idx, (hand_landmarks, handedness) in enumerate(
            zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness)
        ):
            # Determine if left or right hand
            is_right = handedness.classification[0].label == "Right"
            hand_slot = 1 if is_right else 0
            offset = 63 if is_right else 0  # 21 joints × 3 coords

            coords = np.zeros((21, 3), dtype=np.float32)
            
            # Extract landmarks
            for joint_idx, landmark in enumerate(hand_landmarks.landmark):
                coords[joint_idx, 0] = landmark.x
                coords[joint_idx, 1] = landmark.y
                coords[joint_idx, 2] = landmark.z

            wrist = coords[0:1]
            rel_coords = coords - wrist

            landmarks[offset:offset + 63] = rel_coords.reshape(-1)
            wrist_anchor[hand_slot] = wrist[0]
            hand_present[hand_slot] = True

    return landmarks, wrist_anchor, hand_present


def extract_object_pose(image_h, image_w):
    """
    Placeholder for object pose extraction (4×4 → 16-dim).
    In a real scenario, you'd use ArUco markers or object detection.
    
    For now: return zero vector (identity-like)
    """
    # Could implement object detection here
    obj_pose = np.zeros((16,), dtype=np.float32)
    obj_pose[0] = 1.0  # Identity matrix diagonal
    obj_pose[5] = 1.0
    obj_pose[10] = 1.0
    obj_pose[15] = 1.0
    return obj_pose


def suppress_ghost_outliers(
    ghost_rel: np.ndarray,
    measured_rel: np.ndarray,
    max_radius_scale: float,
    abs_radius_cap: float,
) -> np.ndarray:
    """
    Suppress implausible ghost joints by comparing distance-to-wrist against
    the currently measured hand skeleton.

    Both inputs are wrist-relative joint coordinates with shape (21, 3).
    """
    out = ghost_rel.copy()
    ghost_radius = np.linalg.norm(ghost_rel, axis=1)
    measured_radius = np.linalg.norm(measured_rel, axis=1)

    allowed_radius = np.minimum(abs_radius_cap, measured_radius * max_radius_scale + 1e-4)
    invalid = ghost_radius > allowed_radius
    invalid[0] = False  # keep wrist untouched

    out[invalid] = measured_rel[invalid]
    return out


# ─────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str, num_classes: int):
    """Load pre-trained IntentFormer model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer architecture from checkpoint if available
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state = checkpoint['model']
        # Infer d_model from cls_token shape
        d_model = model_state['cls_token'].shape[-1]
        # Infer num_layers by counting encoder layers
        num_layers = sum(1 for k in model_state.keys() if k.startswith('encoder.layers.')) // 12
        # Infer dim_feedforward from first layer
        dim_feedforward = model_state['encoder.layers.0.linear1.weight'].shape[0]
        
        print(f"[✓] Inferred architecture: d_model={d_model}, num_layers={num_layers}, dim_feedforward={dim_feedforward}")
    else:
        # Fallback to default values
        d_model = 128
        num_layers = 4
        dim_feedforward = 512
    
    model = IntentFormer(
        input_dim=378 + 16, # (126*3) hand + 16 obj
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        window_size=30,
        dropout=0.1,
    )
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"[✓] Model loaded from {checkpoint_path}")
    return model


# ─────────────────────────────────────────────────────────
# Real-Time Inference Loop
# ─────────────────────────────────────────────────────────

def run_realtime_inference(args):
    """Main inference loop with webcam and visualization."""
    
    # Device setup
    if torch.backends.mps.is_available() and args.device == "mps":
        device = torch.device("mps")
        print("[✓] Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
        print(f"[✓] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[⚠] Using CPU (this will be slow)")
    
    # Determine number of classes
    num_classes = 3 if args.fusion == "shared_head" else 36
    intent_labels = INTENT_LABELS_SHARED_HEAD if args.fusion == "shared_head" else INTENT_LABELS_CONCAT
    
    # Load model
    model = load_model(args.checkpoint, str(device), num_classes)
    
    model_window_size = int(model.pos_enc.pe.shape[1] - 1)
    if args.window_size != model_window_size:
        print(
            f"[⚠] window_size mismatch: requested={args.window_size}, "
            f"model={model_window_size}. Using model window size."
        )
    window_size = model_window_size

    # Open webcam
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("[ERROR] Failed to open webcam")
        return
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[✓] Camera: {frame_width}x{frame_height} @ {fps} FPS")
    
    # Initialize MediaPipe hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        
        # Sliding window buffer
        landmark_buffer = deque(maxlen=window_size)
        ghost_rel_ema = [None, None]
        
        # FPS counters
        frame_count = 0
        fps_start_time = time.time()
        inference_times = deque(maxlen=30)
        
        print("\n[🎥] Starting real-time inference...")
        print("[💡] Press 'q' to quit, 'p' to pause, 'r' to reset buffer\n")
        
        paused = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run MediaPipe detection
            results = hands.process(rgb_frame)
            
            # Extract hand landmarks (126-dim) and object pose (16-dim)
            hand_landmarks, wrist_anchor, hand_present = extract_hand_landmarks(h, w, results)
            object_pose = extract_object_pose(h, w)
            
            # Add to sliding window
            if not paused:
                combined_features = np.concatenate([hand_landmarks, object_pose])
                landmark_buffer.append(combined_features)
            
            # Perform inference if we have sufficient frames
            inference_msg = ""
            predicted_label = -1
            confidence = 0.0
            ghost_hand = None
            
            if len(landmark_buffer) >= window_size:
                with torch.no_grad():
                    # 1. Prepare Kinematic Features (T, 378)
                    window_data = np.array(list(landmark_buffer))  # (T, 142)
                    pos_history = window_data[:, :126]            # (T, 126)
                    obj_history = window_data[:, 126:]            # (T, 16)
                    
                    # Compute Velocity: v[t] = x[t] - x[t-1]
                    vel_history = np.zeros_like(pos_history)
                    vel_history[1:] = np.diff(pos_history, axis=0)
                    
                    # Compute Acceleration: a[t] = v[t] - v[t-1]
                    acc_history = np.zeros_like(vel_history)
                    acc_history[1:] = np.diff(vel_history, axis=0)
                    
                    # Fuse: (T, 378)
                    kinematic_data = np.concatenate([pos_history, vel_history, acc_history], axis=-1)
                    
                    hand_tensor = torch.from_numpy(kinematic_data).unsqueeze(0).to(device)  # (1, T, 378)
                    obj_tensor = torch.from_numpy(obj_history).unsqueeze(0).to(device)      # (1, T, 16)
                    
                    # Observation ratio
                    obs_ratio = torch.tensor([args.obs_ratio], device=device)
                    
                    # Measure inference time
                    t0 = time.time()
                    logits, pose_pred = model(hand_tensor, obj_tensor, obs_ratio)
                    t_inference = time.time() - t0
                    inference_times.append(t_inference * 1000)  # ms
                    
                    probs = torch.softmax(logits, dim=-1)  # (1, num_classes)
                    
                    predicted_label = logits.argmax(dim=-1).item()
                    confidence = probs.max(dim=-1).values.item()
                    
                    # Extract ghost hand from pose prediction
                    ghost_hand = pose_pred[0].cpu().numpy()  # (126,)
                    
                    inference_msg = f"Intent: {intent_labels[predicted_label]} ({confidence*100:.1f}%)"
            
            # Draw original hand skeleton
            if results.multi_hand_landmarks:
                for hand_landmarks_mp, handedness in zip(
                    results.multi_hand_landmarks, 
                    results.multi_handedness
                ):
                    is_right = handedness.classification[0].label == "Right"
                    color = (255, 0, 0) if is_right else (0, 0, 255)
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks_mp,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
            
            # Draw ghost hand (predicted next pose)
            if ghost_hand is not None and len(landmark_buffer) >= window_size:
                for hand_idx in range(2):
                    if not hand_present[hand_idx]:
                        ghost_rel_ema[hand_idx] = None

                # Reshape predicted pose to (21, 3) per hand
                for hand_idx in range(2):
                    if not hand_present[hand_idx]:
                        continue

                    offset = hand_idx * 63  # 21 joints × 3 coords
                    ghost_rel = ghost_hand[offset:offset+63].reshape(21, 3)
                    measured_rel = hand_landmarks[offset:offset+63].reshape(21, 3)

                    ghost_rel = suppress_ghost_outliers(
                        ghost_rel=ghost_rel,
                        measured_rel=measured_rel,
                        max_radius_scale=args.ghost_max_radius_scale,
                        abs_radius_cap=args.ghost_abs_radius_cap,
                    )

                    if ghost_rel_ema[hand_idx] is None:
                        ghost_rel_ema[hand_idx] = ghost_rel
                    else:
                        alpha = args.ghost_smooth_alpha
                        ghost_rel_ema[hand_idx] = alpha * ghost_rel + (1.0 - alpha) * ghost_rel_ema[hand_idx]

                    ghost_joints = ghost_rel_ema[hand_idx] + wrist_anchor[hand_idx:hand_idx+1]
                    
                    # Draw ghost skeleton
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start, end = connection
                        x1 = int(ghost_joints[start, 0] * w) if 0 < ghost_joints[start, 0] < 1 else -1
                        y1 = int(ghost_joints[start, 1] * h) if 0 < ghost_joints[start, 1] < 1 else -1
                        x2 = int(ghost_joints[end, 0] * w) if 0 < ghost_joints[end, 0] < 1 else -1
                        y2 = int(ghost_joints[end, 1] * h) if 0 < ghost_joints[end, 1] < 1 else -1
                        
                        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                            cv2.line(frame, (x1, y1), (x2, y2), COLOR_GHOST, 2)
                    
                    # Draw ghost joints
                    for joint_idx, joint in enumerate(ghost_joints):
                        if 0 < joint[0] < 1 and 0 < joint[1] < 1:
                            x = int(joint[0] * w)
                            y = int(joint[1] * h)
                            cv2.circle(frame, (x, y), 3, COLOR_GHOST, -1)
            
            # Render text info
            cv2.putText(
                frame,
                f"Buffer: {len(landmark_buffer)}/{window_size}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            
            if inference_msg:
                color = COLOR_CONFIDENT if confidence > 0.7 else (COLOR_UNCERTAIN if confidence > 0.5 else COLOR_LOW_CONFIDENCE)
                cv2.putText(
                    frame,
                    inference_msg,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            
            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                actual_fps = frame_count / elapsed
                cv2.putText(
                    frame,
                    f"FPS: {actual_fps:.1f} | Inference: {np.mean(list(inference_times)) if inference_times else 0:.1f}ms",
                    (10, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                )
                frame_count = 0
                fps_start_time = time.time()
            
            # Status line
            status = "[PAUSED]" if paused else "[RUNNING]"
            cv2.putText(
                frame,
                status,
                (frame_width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255) if paused else (0, 255, 0),
                2,
            )
            
            # Display frame
            cv2.imshow("IntentFormer Real-Time Inference", frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[🛑] Exiting...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"[{'⏸' if paused else '▶'}] {'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):
                landmark_buffer.clear()
                ghost_rel_ema = [None, None]
                print("[🔄] Buffer reset")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Inference session closed")


# ─────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time Intent Prediction with Ghost Hand Visualization"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/combined_best/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="shared_head",
        choices=["shared_head", "concat"],
        help="Fusion type (determines number of output classes)",
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="Requested temporal window size (will auto-align with model if needed)",
    )
    parser.add_argument(
        "--obs_ratio",
        type=float,
        default=0.30,
        help="Observation ratio fed to model (recommended: 0.2, 0.25, or 0.3)",
    )
    parser.add_argument(
        "--ghost_smooth_alpha",
        type=float,
        default=0.35,
        help="EMA alpha for ghost smoothing (0=very smooth, 1=no smoothing)",
    )
    parser.add_argument(
        "--ghost_max_radius_scale",
        type=float,
        default=1.8,
        help="Max allowed ghost joint radius vs measured radius",
    )
    parser.add_argument(
        "--ghost_abs_radius_cap",
        type=float,
        default=0.45,
        help="Absolute cap for wrist-relative ghost joint radius",
    )
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not (0.0 < args.obs_ratio <= 1.0):
        print("[ERROR] --obs_ratio must be in (0, 1].")
        sys.exit(1)

    if not (0.0 < args.ghost_smooth_alpha <= 1.0):
        print("[ERROR] --ghost_smooth_alpha must be in (0, 1].")
        sys.exit(1)

    if args.ghost_max_radius_scale <= 0.0:
        print("[ERROR] --ghost_max_radius_scale must be > 0.")
        sys.exit(1)

    if args.ghost_abs_radius_cap <= 0.0:
        print("[ERROR] --ghost_abs_radius_cap must be > 0.")
        sys.exit(1)
    
    run_realtime_inference(args)


if __name__ == "__main__":
    main()
