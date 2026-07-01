import numpy as np
import json
from pathlib import Path

print("=" * 80)
print("OAKINK DATASET ANALYSIS")
print("=" * 80)

oakink_path = Path("/Users/muratcelik/Desktop/Thesis/Workspace/AuraXR/data/processed/oakink_canonical")
dataset_npz = oakink_path / "dataset.npz"
split_path = oakink_path / "split.json"

# Load OakInk data
data = np.load(dataset_npz, allow_pickle=True)
print(f"\nOakInk dataset.npz keys: {data.files}")
print(f"pose shape: {data['pose'].shape}")
print(f"shape shape: {data['shape'].shape}")
print(f"tsl shape: {data['tsl'].shape}")

# Check for distance field
if 'dist' in data.files:
    dist = data['dist']
    print(f"\ndist field found, shape: {dist.shape}")
    print(f"dist min: {dist.min():.6f}, max: {dist.max():.6f}, mean: {dist.mean():.6f}")
    print(f"dist percentiles [10,25,50,75,90]: {np.percentile(dist, [10,25,50,75,90])}")
else:
    print("\nNo 'dist' field directly in dataset.npz")
    print("Note: OakInk computes distance at runtime from object geometry + wrist translation")

# Load split info
with open(split_path) as f:
    splits = json.load(f)

total_samples = 0
for split_name, indices in splits.items():
    print(f"\n{split_name}: {len(indices)} samples")
    total_samples += len(indices)

print(f"\nTotal OakInk samples: {total_samples}")

# Check for contact_flag
if 'contact_flag' in data.files:
    contact = data['contact_flag']
    print(f"\ncontact_flag found, shape: {contact.shape}")
    print(f"contact_flag=1: {(contact == 1).sum()} / {len(contact)} = {100 * (contact == 1).sum() / len(contact):.1f}%")
else:
    print("\nNo contact_flag in OakInk dataset.npz")

print("\n" + "=" * 80)
print("HOT3D DATASET ANALYSIS")
print("=" * 80)

hot3d_path = Path("/Users/muratcelik/Desktop/Thesis/Workspace/AuraXR/data/processed/hot3d_canonical")
manifest_path = hot3d_path / "manifest.csv"
obj_split_path = hot3d_path / "obj_split.json"

# Count total frames and analyze distance distribution
all_dists = []
all_contacts = []
seq_files = sorted(hot3d_path.glob("seq_*.npz"))
print(f"\nFound {len(seq_files)} sequence files")

split_counts = {}
for seq_file in seq_files[:5]:  # Sample first 5 sequences
    data = np.load(seq_file, allow_pickle=True)
    print(f"\n{seq_file.name}:")
    print(f"  keys: {data.files}")
    print(f"  n_frames: {len(data['finger_aa45'])}")
    
    if 'dist' in data.files:
        dist = data['dist']
        print(f"  dist shape: {dist.shape}")
        if len(dist.shape) > 1:
            dist = dist.squeeze()
        print(f"  dist min: {dist.min():.6f}, max: {dist.max():.6f}, mean: {dist.mean():.6f}")
        all_dists.extend(dist.flatten())
    
    if 'contact_flag' in data.files:
        contact = data['contact_flag']
        print(f"  contact_flag shape: {contact.shape}, sum: {contact.sum()}")
        all_contacts.extend(contact.flatten())

# Aggregate HOT3D distance stats
if all_dists:
    all_dists = np.array(all_dists)
    print(f"\n\nHOT3D Distance Stats (from first 5 sequences):")
    print(f"  min: {all_dists.min():.6f}, max: {all_dists.max():.6f}, mean: {all_dists.mean():.6f}")
    print(f"  percentiles [10,25,50,75,90]: {np.percentile(all_dists, [10,25,50,75,90])}")

# Count total frames in all sequences
total_hot3d_frames = 0
for seq_file in seq_files:
    data = np.load(seq_file, allow_pickle=True)
    total_hot3d_frames += len(data['finger_aa45'])

print(f"\nTotal HOT3D frames across all sequences: {total_hot3d_frames}")

# Load manifest to understand splits
print("\nManifest splits:")
splits_hot3d = {}
with open(manifest_path) as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        parts = line.strip().split(',')
        if len(parts) >= 2:
            seq_id = parts[0]
            split = parts[1] if len(parts) > 1 else 'unknown'
            splits_hot3d[split] = splits_hot3d.get(split, 0) + 1

for split, count in sorted(splits_hot3d.items()):
    print(f"  {split}: {count} sequences")

