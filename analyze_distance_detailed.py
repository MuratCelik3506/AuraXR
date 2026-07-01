import numpy as np
import json
from pathlib import Path

print("=" * 80)
print("DETAILED DISTANCE & CONTACT ANALYSIS")
print("=" * 80)

# HOT3D: Aggregate all distance data
hot3d_path = Path("/Users/muratcelik/Desktop/Thesis/Workspace/AuraXR/data/processed/hot3d_canonical")
seq_files = sorted(hot3d_path.glob("seq_*.npz"))

all_hot3d_dists = []
all_hot3d_contacts = []
total_hot3d_frames = 0

print("\nProcessing all HOT3D sequences...")
for seq_file in seq_files:
    data = np.load(seq_file, allow_pickle=True)
    n_frames = len(data['finger_aa45'])
    total_hot3d_frames += n_frames
    
    if 'dist' in data.files:
        dist = data['dist'].flatten()
        all_hot3d_dists.extend(dist)
    
    if 'contact_flag' in data.files:
        contact = data['contact_flag'].flatten()
        all_hot3d_contacts.extend(contact)

all_hot3d_dists = np.array(all_hot3d_dists)
all_hot3d_contacts = np.array(all_hot3d_contacts)

print(f"\nHOT3D DATASET SUMMARY:")
print(f"Total sequences: {len(seq_files)}")
print(f"Total frames: {total_hot3d_frames}")
print(f"Total distance samples: {len(all_hot3d_dists)}")
print(f"Total contact samples: {len(all_hot3d_contacts)}")

print(f"\nHOT3D Distance Statistics (in meters):")
print(f"  Min: {all_hot3d_dists.min():.6f} m")
print(f"  Max: {all_hot3d_dists.max():.6f} m")
print(f"  Mean: {all_hot3d_dists.mean():.6f} m")
print(f"  Std: {all_hot3d_dists.std():.6f} m")
print(f"  Median: {np.median(all_hot3d_dists):.6f} m")
print(f"  Percentiles [5,10,25,50,75,90,95]: {np.percentile(all_hot3d_dists, [5,10,25,50,75,90,95])}")

# Distance distribution buckets
print(f"\nHOT3D Distance Distribution:")
print(f"  < 2 mm (0.002 m): {(all_hot3d_dists < 0.002).sum()} / {len(all_hot3d_dists)} = {100 * (all_hot3d_dists < 0.002).sum() / len(all_hot3d_dists):.2f}%")
print(f"  2-30 mm (0.002-0.03 m): {((all_hot3d_dists >= 0.002) & (all_hot3d_dists < 0.03)).sum()} / {len(all_hot3d_dists)} = {100 * ((all_hot3d_dists >= 0.002) & (all_hot3d_dists < 0.03)).sum() / len(all_hot3d_dists):.2f}%")
print(f"  30 mm - 30 cm (0.03-0.30 m): {((all_hot3d_dists >= 0.03) & (all_hot3d_dists < 0.30)).sum()} / {len(all_hot3d_dists)} = {100 * ((all_hot3d_dists >= 0.03) & (all_hot3d_dists < 0.30)).sum() / len(all_hot3d_dists):.2f}%")
print(f"  > 30 cm (> 0.30 m): {(all_hot3d_dists >= 0.30).sum()} / {len(all_hot3d_dists)} = {100 * (all_hot3d_dists >= 0.30).sum() / len(all_hot3d_dists):.2f}%")

print(f"\nHOT3D Contact Flag Statistics:")
print(f"  contact_flag = 1: {(all_hot3d_contacts == 1).sum()} / {len(all_hot3d_contacts)} = {100 * (all_hot3d_contacts == 1).sum() / len(all_hot3d_contacts):.2f}%")
print(f"  contact_flag = 0: {(all_hot3d_contacts == 0).sum()} / {len(all_hot3d_contacts)} = {100 * (all_hot3d_contacts == 0).sum() / len(all_hot3d_contacts):.2f}%")

# Analyze contact vs distance correlation
contact_dists = all_hot3d_dists[all_hot3d_contacts == 1]
no_contact_dists = all_hot3d_dists[all_hot3d_contacts == 0]

print(f"\nHOT3D Contact=1 distance stats:")
print(f"  Mean dist: {contact_dists.mean():.6f} m")
print(f"  Max dist: {contact_dists.max():.6f} m")
print(f"  % < 5cm: {100 * (contact_dists < 0.05).sum() / len(contact_dists):.2f}%")

print(f"\nHOT3D Contact=0 distance stats:")
print(f"  Mean dist: {no_contact_dists.mean():.6f} m")
print(f"  Min dist: {no_contact_dists.min():.6f} m")
print(f"  % < 5cm: {100 * (no_contact_dists < 0.05).sum() / len(no_contact_dists):.2f}%")

print("\n" + "=" * 80)
print("OAKINK DATASET SUMMARY")
print("=" * 80)

oakink_path = Path("/Users/muratcelik/Desktop/Thesis/Workspace/AuraXR/data/processed/oakink_canonical")
dataset_npz = oakink_path / "dataset.npz"
split_path = oakink_path / "split.json"

data = np.load(dataset_npz, allow_pickle=True)

print(f"\nOakInk total samples: {data['pose'].shape[0]}")

# Load split info
with open(split_path) as f:
    splits = json.load(f)

print(f"\nOakInk Splits:")
for split_name in sorted(splits.keys()):
    indices = splits[split_name]
    print(f"  {split_name}: {len(indices)} samples")

print(f"\nNote: OakInk 'dist' is computed at runtime:")
print(f"  - Nearest wrist-to-object-surface distance in meters")
print(f"  - Computed from sampled object point cloud (real geometry)")
print(f"  - Wrist position: tsl (3D translation)")
print(f"  - Object points transformed to world frame using obj_anno (R,t)")

print(f"\nNote: HOT3D 'dist' is:")
print(f"  - Pre-computed distance between wrist and closest object surface")
print(f"  - Stored per-frame in (N,1) array in each seq_*.npz")
print(f"  - Contact appears to occur when dist < ~0.05m (5cm)")

