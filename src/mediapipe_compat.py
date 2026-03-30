"""
Compatibility layer for MediaPipe legacy API (solutions module)
This module provides the old mediapipe.solutions API using cv2 and mediapipe-legacy approach
"""

import cv2
import numpy as np

# For newer mediapipe, we need to use opencv instead
try:
    import mediapipe as mp
    # Check if old API exists
    if hasattr(mp, 'solutions'):
        # Old API is available, use it directly
        hands = mp.solutions.hands
        drawing_utils = mp.solutions.drawing_utils
        drawing_styles = mp.solutions.drawing_styles
    else:
        # New API - create compatibility wrapper
        # Unfortunately, the new API structure is completely different
        # We'll need to use opencv-based hand detection or install mediapipe-legacy
        raise ImportError("MediaPipe solutions API not available. Please install an older version of mediapipe or mediapipe-model-maker")
except Exception as e:
    raise ImportError(f"Failed to import MediaPipe compatibility: {e}")
