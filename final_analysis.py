import numpy as np
from pathlib import Path

print("=" * 80)
print("FINAL DISTANCE DISTRIBUTION ANALYSIS")
print("=" * 80)

# HOT3D aggregation
hot3d_path = Path("/Users/muratcelik/Desktop/Thesis/Workspace/AuraXR/data/processed/hot3d_canonical")
seq_files = sorted(hot3d_path.glob("seq_*.npz"))
all_hot3d_dists = []
all_hot3d_contacts = []

for seq_file in seq_files:
    data = np.load(seq_file, allow_pickle=True)
    if 'dist' in data.files:
        all_hot3d_dists.extend(data['dist'].flatten())
    if 'contact_flag' in data.files:
        all_hot3d_contacts.extend(data['contact_flag'].flatten())

all_hot3d_dists = np.array(all_hot3d_dists)
all_hot3d_contacts = np.array(all_hot3d_contacts)

print("\nHOT3D:")
print(f"  Total samples: {len(all_hot3d_dists)}")
print(f"  Distance field: Pre-computed wrist-to-object-surface (meters)")
print(f"  Distance range: {all_hot3d_dists.min():.6f} - {all_hot3d_dists.max():.6f} m")
print(f"  Distance mean/std: {all_hot3d_dists.mean():.6f} ± {all_hot3d_dists.std():.6f} m")
print(f"  ")
print(f"  Distribution by distance threshold:")
print(f"    < 2 mm (< 0.002 m): {(all_hot3d_dists < 0.002).sum():6d} samples ({100*(all_hot3d_dists < 0.002).sum()/len(all_hot3d_dists):5.2f}%)")
print(f"    2-30 mm (0.002-0.03 m): {((all_hot3d_dists >= 0.002) & (all_hot3d_dists < 0.03)).sum():6d} samples ({100*((all_hot3d_dists >= 0.002) & (all_hot3d_dists < 0.03)).sum()/len(all_hot3d_dists):5.2f}%)")
print(f"    30 mm - 30 cm (0.03-0.30 m): {((all_hot3d_dists >= 0.03) & (all_hot3d_dists < 0.30)).sum():6d} samples ({100*((all_hot3d_dists >= 0.03) & (all_hot3d_dists < 0.30)).sum()/len(all_hot3d_dists):5.2f}%)")
print(f"    > 30 cm (> 0.30 m): {(all_hot3d_dists >= 0.30).sum():6d} samples ({100*(all_hot3d_dists >= 0.30).sum()/len(all_hot3d_dists):5.2f}%)")
print(f"  ")
print(f"  Contact flag (threshold: dist < 0.03 m):")
print(f"    contact_flag = 1: {(all_hot3d_contacts == 1).sum():6d} samples ({100*(all_hot3d_contacts == 1).sum()/len(all_hot3d_contacts):5.2f}%)")
print(f"    contact_flag = 0: {(all_hot3d_contacts == 0).sum():6d} samples ({100*(all_hot3d_contacts == 0).sum()/len(all_hot3d_contacts):5.2f}%)")

print(f"\nOakInk:")
oakink_path = Path("/Users/muratcelik/Desktop/Thesis/Workspace/AuraXR/data/processed/oakink_canonical")
data = np.load(oakink_path / "dataset.npz", allow_pickle=True)
import json
with open(oakink_path / "split.json") as f:
    splits = json.load(f)

total_samples = sum(len(indices) for indices in splits.values())
print(f"  Total samples: {total_samples} (unique: {data['pose'].shape[0]})")
print(f"  Distance field: Runtime-computed nearest fingertip-independent wrist-to-surface (meters)")
print(f"  Distance computation: min(||obj_pts_world - wrist_tsl||)")
print(f"    where obj_pts_world = obj_pts_canonical @ R_obj.T + t_obj (from obj_anno)")
print(f"  ")
print(f"  Splits:")
for split_name in sorted(splits.keys()):
    indices = splits[split_name]
    print(f"    {split_name:15s}: {len(indices):5d} indices")
print(f"  ")
print(f"  Note: No contact_flag in OakInk. Static single-frame dataset.")
print(f"  Note: rel_vel is zero-filled (no temporal window)")

