"""grip_categories.py — Maps HOT3D BOP object IDs (1-33) to grip category and bbox half-extents.

Run: python grip_categories.py   (prints self-test table)
"""

import numpy as np

# Grip category indices
POWER     = 0   # cylindrical wrap — cups, bottles, cans
PRECISION = 1   # fingertip — thin/long tools
PALMAR    = 2   # flat palm — plates, phones, remotes
PINCH     = 3   # pinch/small — mouse, puzzle, small objects

GRIP_NAMES = {POWER: "Power", PRECISION: "Precision", PALMAR: "Palmar", PINCH: "Pinch"}

# BOP ID → (grip_category, bbox_half_extents_xyz_m, name)
# bbox half-extents: half of bounding box in each axis (meters)
# Approximate real-world dimensions measured from object references.
OBJ_INFO = {
    1:  (PINCH,     (0.030, 0.075, 0.030), "holder_black"),
    2:  (POWER,     (0.100, 0.040, 0.100), "bowl"),
    3:  (PALMAR,    (0.130, 0.010, 0.130), "plate_bamboo"),
    4:  (PRECISION, (0.015, 0.010, 0.150), "spoon_wooden"),
    5:  (PRECISION, (0.045, 0.010, 0.125), "potato_masher"),
    6:  (PRECISION, (0.035, 0.005, 0.140), "spatula_red"),
    7:  (POWER,     (0.075, 0.125, 0.060), "coffee_pot"),
    8:  (POWER,     (0.045, 0.050, 0.045), "mug_patterned"),
    9:  (POWER,     (0.045, 0.050, 0.045), "mug_white"),
    10: (POWER,     (0.038, 0.050, 0.038), "can_soup"),
    11: (POWER,     (0.045, 0.075, 0.045), "can_parmesan"),
    12: (POWER,     (0.038, 0.055, 0.038), "can_tomato_sauce"),
    13: (POWER,     (0.035, 0.100, 0.035), "bottle_mustard"),
    14: (POWER,     (0.035, 0.140, 0.035), "bottle_bbq"),
    15: (POWER,     (0.030, 0.140, 0.030), "bottle_ranch"),
    16: (POWER,     (0.060, 0.150, 0.060), "vase"),
    17: (POWER,     (0.045, 0.105, 0.045), "carton_milk"),
    18: (POWER,     (0.045, 0.100, 0.045), "carton_oj"),
    19: (POWER,     (0.035, 0.090, 0.035), "flask"),
    20: (PALMAR,    (0.100, 0.020, 0.075), "food_waffles"),
    21: (POWER,     (0.050, 0.060, 0.050), "food_vegetables"),
    22: (POWER,     (0.125, 0.050, 0.050), "dumbbell_5lb"),
    23: (PINCH,     (0.075, 0.025, 0.040), "aria_small"),
    24: (PALMAR,    (0.040, 0.005, 0.080), "cellphone"),
    25: (PINCH,     (0.030, 0.075, 0.030), "holder_gray"),
    26: (POWER,     (0.075, 0.090, 0.075), "birdhouse_toy"),
    27: (PINCH,     (0.060, 0.050, 0.030), "dino_toy"),
    28: (PALMAR,    (0.175, 0.015, 0.075), "keyboard"),
    29: (PALMAR,    (0.060, 0.020, 0.030), "whiteboard_eraser"),
    30: (PINCH,     (0.075, 0.020, 0.075), "puzzle_toy"),
    31: (PINCH,     (0.030, 0.020, 0.050), "mouse"),
    32: (PRECISION, (0.008, 0.008, 0.070), "whiteboard_marker"),
    33: (PALMAR,    (0.025, 0.008, 0.090), "dvd_remote"),
}

# Quick-lookup arrays derived from OBJ_INFO
OBJ_GRIP   = {bop_id: info[0] for bop_id, info in OBJ_INFO.items()}
OBJ_BBOX   = {bop_id: np.array(info[1], dtype=np.float32) for bop_id, info in OBJ_INFO.items()}
OBJ_NAMES  = {bop_id: info[2] for bop_id, info in OBJ_INFO.items()}


def grip_onehot(bop_id: int) -> np.ndarray:
    """Return 4-dim one-hot vector for the grip category of the given BOP object ID."""
    cat = OBJ_GRIP.get(bop_id, PINCH)
    oh = np.zeros(4, dtype=np.float32)
    oh[cat] = 1.0
    return oh


def object_features(bop_id: int):
    """Return (grip_onehot(4), bbox(3)) — the 7-dim object branch input.

    Falls back to PINCH + zero bbox for unknown IDs.
    """
    oh   = grip_onehot(bop_id)
    bbox = OBJ_BBOX.get(bop_id, np.zeros(3, dtype=np.float32))
    return oh, bbox


if __name__ == "__main__":
    print(f"\n{'BOP':>5}  {'Name':<22}  {'Grip':<12}  {'BBox half-extents (m)':>28}")
    print(f"{'-----':>5}  {'----------------------':<22}  {'------------':<12}  {'----------------------------':>28}")
    for bop_id in sorted(OBJ_INFO):
        cat, bbox, name = OBJ_INFO[bop_id]
        cat_name = GRIP_NAMES[cat]
        bbox_str = f"[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}]"
        print(f"{bop_id:>5}  {name:<22}  {cat_name:<12}  {bbox_str:>28}")

    print(f"\nGrip category counts:")
    from collections import Counter
    counts = Counter(v[0] for v in OBJ_INFO.values())
    for cat, n in sorted(counts.items()):
        print(f"  {GRIP_NAMES[cat]:<12} ({cat}): {n} objects")

    print("\nSelf-test:")
    oh, bbox = object_features(9)
    assert oh[POWER] == 1.0 and oh.sum() == 1.0, "mug_white should be Power"
    oh, bbox = object_features(4)
    assert oh[PRECISION] == 1.0, "spoon_wooden should be Precision"
    oh, bbox = object_features(3)
    assert oh[PALMAR] == 1.0, "plate_bamboo should be Palmar"
    oh, bbox = object_features(31)
    assert oh[PINCH] == 1.0, "mouse should be Pinch"
    print("  All assertions passed.")
