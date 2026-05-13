# HOT3D Exploration Scripts

Activate the virtual environment before running any script:

```bash
cd hot3d_exploration
source .venv/bin/activate
```

## Scripts — run in order

| Script | What it does | Run when |
|--------|-------------|----------|
| `00_check_env.py` | Verify all packages are installed | First time, after any install |
| `01_explore_clips.py` | Inspect raw clip structure — keys, images, metadata | Before anything else |
| `02_inspect_mano.py` | Analyse MANO annotation schema, β distribution, bimanual rate | Answering Q-E, Q-F |
| `03_controller_proxy.py` | Derive synthetic controller poses from wrist transforms | Answering Q-A |
| `04_fps_temporal_window.py` | Measure actual frame rate, analyse temporal window options | Answering Q-C |
| `05_object_categories.py` | Catalogue object categories, recommend POC subset | Answering Q1 in plan |

## Quick start

```bash
# Check everything is installed
python 00_check_env.py

# Inspect first clip (no HF auth needed for public data)
python 01_explore_clips.py --n 1

# Check MANO annotations over 5 clips
python 02_inspect_mano.py --n_clips 5 --plot

# Analyse controller proxy strategy
python 03_controller_proxy.py --n_clips 3 --plot

# Check frame rate
python 04_fps_temporal_window.py --n_clips 5

# Catalogue objects (run with more clips for full picture)
python 05_object_categories.py --n_clips 50 --plot
```

## HuggingFace Auth (if needed)

```bash
.venv/bin/huggingface-cli login
# Paste your HF token — get one at huggingface.co/settings/tokens
```

## Optional packages (install when needed)

```bash
# MANO forward kinematics (requires MANO license at mano.is.tue.mpg.de)
pip install smplx

# Aria / VRS format (full HOT3D dataset, not clips)
pip install projectaria_tools

# 3D visualiser
pip install rerun-sdk
```
