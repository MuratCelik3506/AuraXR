"""
00_check_env.py — Verify all required packages are installed and importable.
Run this before any other script to confirm the environment is ready.
"""

import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("datasets", "datasets"),
    ("webdataset", "webdataset"),
    ("trimesh", "trimesh"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("pandas", "pandas"),
    ("huggingface_hub", "huggingface_hub"),
]

OPTIONAL = [
    ("projectaria_tools", "projectaria_tools"),
    ("smplx", "smplx"),
    ("rerun", "rerun-sdk"),
    ("open3d", "open3d"),
]


def check(packages, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    all_ok = True
    for module, pkg in packages:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "unknown")
            print(f"  [OK]      {pkg:<30} version={version}")
        except ImportError:
            print(f"  [MISSING] {pkg:<30} → pip install {pkg}")
            all_ok = False
    return all_ok


required_ok = check(REQUIRED, "REQUIRED packages")
optional_ok = check(OPTIONAL, "OPTIONAL packages (install when needed)")

print(f"\n{'='*50}")
print(f"  Python: {sys.version}")
print(f"  Required: {'ALL OK' if required_ok else 'SOME MISSING — see above'}")
print(f"  Optional: {'ALL OK' if optional_ok else 'some missing (ok for now)'}")
print(f"{'='*50}\n")

if not required_ok:
    sys.exit(1)
