"""mano_fk.py — MANO forward kinematics wrapper for AuraXR evaluation.

Wraps smplx MANO model to convert:
  pose_15  (N, 15)  — MANO PCA hand pose components (HOT3D format)
  wrist_6d (N, 6)   — 6D wrist rotation (same as existing pipeline)
  → joints_3d (N, 21, 3)  — 3D joint positions in mm, wrist-relative

MANO weights required at: data/models/mano/MANO_LEFT.pkl / MANO_RIGHT.pkl
Download (free registration): https://mano.is.tue.mpg.de

PCA basis export (for Unity MANODecoder.cs):
  python src/mano_fk.py --export_pca data/models/mano/pca_left.json left
  python src/mano_fk.py --export_pca data/models/mano/pca_right.json right
"""

from __future__ import annotations
import json
import sys
import pickle
import numpy as np
import torch
from pathlib import Path


MANO_MODELS_DIR = Path(__file__).parent.parent / "data" / "models" / "mano"
NUM_PCA_COMPS = 15


# ── chumpy compatibility shim ─────────────────────────────────────────────────
# MANO pkl files were saved with chumpy arrays. chumpy is incompatible with
# Python 3.10+. Patch pickle to silently convert chumpy types to ndarray.
class _ChumShim(np.ndarray):
    pass

class _ChumModule(sys.modules.get("types", None).__class__ if False else object):
    pass

def _make_chumpy_safe_loader():
    """Return a drop-in replacement for pickle.load that converts chumpy objects to ndarray.

    MANO pkl files were saved with chumpy arrays. We use a custom Unpickler
    that maps every chumpy class to a np.ndarray subclass so smplx can use
    the data directly without chumpy being installed.
    """
    import pickle as _pkl
    import io, types

    class _NdArraySubclass(np.ndarray):
        """Minimal numpy array subclass that accepts any chumpy __setstate__."""
        def __new__(cls, *a, **kw):
            return np.array([]).view(cls)
        def __setstate__(self, state):
            if isinstance(state, dict):
                # Typical chumpy state: has 'x' (the raw array data)
                x = state.get("x", state.get("_array", None))
                if x is not None:
                    try:
                        arr = np.asarray(x)
                        # Resize self in-place to hold the data
                        self.resize(arr.shape, refcheck=False)
                        self[:] = arr
                    except Exception:
                        pass
            elif isinstance(state, (list, tuple)) and len(state) >= 5:
                # numpy __setstate__ format: (version, shape, dtype, fortran, data)
                super().__setstate__(state)

    class _ChumUnpickler(_pkl.Unpickler):
        """Redirect all chumpy.* class lookups to _NdArraySubclass."""
        def find_class(self, module, name):
            if module.startswith("chumpy"):
                return _NdArraySubclass
            return super().find_class(module, name)

    def safe_load(f, encoding="latin1"):
        data = f.read()
        return _ChumUnpickler(io.BytesIO(data), encoding=encoding).load()  # type: ignore[call-arg]

    return safe_load


def _patch_smplx_pickle():
    """Monkey-patch smplx body_models to use our chumpy-safe loader."""
    try:
        import smplx.body_models as _bm
        import pickle as _pkl
        _safe = _make_chumpy_safe_loader()
        _bm.pickle = type(_pkl)(  # type: ignore[attr-defined]
            "pickle_patched",
        )
        import types as _t
        patched = _t.ModuleType("pickle_patched")
        patched.__dict__.update(_pkl.__dict__)
        patched.load = _safe
        _bm.pickle = patched
    except Exception as e:
        print(f"[mano_fk] chumpy patch warning: {e}")

_patch_smplx_pickle()


# ── Rotation utilities ────────────────────────────────────────────────────────

def rot6d_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """6D rotation → (N, 3, 3) rotation matrix via Gram-Schmidt."""
    a1 = r[:, :3]
    a2 = r[:, 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)   # (N, 3, 3)


def rot6d_to_matrix_np(r: np.ndarray) -> np.ndarray:
    """Numpy version of rot6d_to_matrix. r: (N, 6) → (N, 3, 3)."""
    a1 = r[:, :3]
    a2 = r[:, 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)      # (N, 3, 3)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) rotation matrices → (N, 3) axis-angle."""
    # Uses Rodrigues via trace
    batch = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))  # (N,)
    # Skew-symmetric part
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)  # (N, 3)
    sin_theta = torch.sin(theta).unsqueeze(-1).clamp(min=1e-7)
    axis = axis / (2 * sin_theta)
    return axis * theta.unsqueeze(-1)


# ── MANO FK wrapper ───────────────────────────────────────────────────────────

class MANOForwardKinematics:
    """Thin wrapper around smplx MANO for AuraXR evaluation.

    Usage:
        fk = MANOForwardKinematics('left')
        joints = fk(pose_15, wrist_6d)  # (N,21,3) in mm, wrist-relative
    """

    def __init__(self, hand: str = "right", device: str | None = None):
        assert hand in ("left", "right"), "hand must be 'left' or 'right'"
        self.hand = hand
        # Force CPU for MANO FK — MPS has allocation issues with smplx sparse tensors
        self.device = torch.device("cpu")
        self._model = None   # lazy-loaded

    def _load(self):
        """Load MANO model on first use. Raises FileNotFoundError if weights missing."""
        import smplx
        pkl = MANO_MODELS_DIR / f"MANO_{'RIGHT' if self.hand == 'right' else 'LEFT'}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(
                f"MANO weights not found at {pkl}.\n"
                "Download from https://mano.is.tue.mpg.de (free registration) and place\n"
                f"MANO_LEFT.pkl + MANO_RIGHT.pkl in {MANO_MODELS_DIR}/"
            )
        self._model = smplx.create(
            str(MANO_MODELS_DIR),
            model_type="mano",
            is_rhand=(self.hand == "right"),
            use_pca=True,
            num_pca_comps=NUM_PCA_COMPS,
            flat_hand_mean=False,
        ).to(self.device).eval()

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @torch.no_grad()
    def __call__(
        self,
        pose_15: np.ndarray | torch.Tensor,
        wrist_6d: np.ndarray | torch.Tensor,
        betas: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        pose_15  : (N, 15) MANO PCA pose components
        wrist_6d : (N, 6)  6D wrist rotation
        betas    : (N, 10) or None → uses zero betas (canonical shape)
        Returns  : (N, 21, 3) joint positions in metres, wrist at origin
        """
        N = pose_15.shape[0] if hasattr(pose_15, "shape") else len(pose_15)

        def to_t(x, dim):
            if x is None:
                return torch.zeros(N, dim, device=self.device, dtype=torch.float32)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            return x.float().to(self.device)

        p15  = to_t(pose_15,  NUM_PCA_COMPS)
        w6d  = to_t(wrist_6d, 6)
        bts  = to_t(betas, 10)

        global_rot = rot6d_to_matrix(w6d)             # (N, 3, 3)
        global_aa  = matrix_to_axis_angle(global_rot) # (N, 3)

        out = self.model(
            hand_pose=p15,
            global_orient=global_aa,
            betas=bts,
        )
        # joints: (N, 21, 3) in metres; subtract joint[0] (wrist) to make wrist-relative
        joints = out.joints  # (N, 21, 3)
        joints = joints - joints[:, :1, :]             # wrist-relative
        return joints.cpu().numpy()                    # (N, 21, 3) metres

    def joints_to_mm(self, joints: np.ndarray) -> np.ndarray:
        """Convert metre joints to millimetres."""
        return joints * 1000.0

    def pca_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (pca_matrix, mean_pose) for Unity MANODecoder.
        pca_matrix : (15, 45)  — rows = PCA components in axis-angle space
        mean_pose  : (45,)     — mean hand pose (axis-angle)
        """
        m = self.model
        basis = m.hand_components.detach().cpu().numpy()   # (15, 45)
        mean  = m.hand_mean.detach().cpu().numpy()         # (45,)
        return basis, mean


# ── MPJPE helpers ─────────────────────────────────────────────────────────────

def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error in millimetres.
    pred, gt: (N, J, 3) in metres (converted internally to mm).
    """
    diff = (pred - gt) * 1000.0   # → mm
    return float(np.linalg.norm(diff, axis=-1).mean())


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-aligned MPJPE in mm.
    Removes global rotation/translation/scale before computing error.
    """
    from scipy.spatial.transform import Rotation

    N, J, _ = pred.shape
    errors = []
    for i in range(N):
        p = pred[i]   # (J, 3)
        g = gt[i]

        # Centre both
        p_c = p - p.mean(0)
        g_c = g - g.mean(0)

        # Optimal rotation (SVD)
        H = p_c.T @ g_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        # Scale
        scale = (g_c * (p_c @ R.T)).sum() / (p_c ** 2).sum()
        p_aligned = scale * (p_c @ R.T) + g.mean(0)
        errors.append(np.linalg.norm((p_aligned - g) * 1000.0, axis=-1).mean())

    return float(np.mean(errors))


# ── CLI: PCA export for Unity ─────────────────────────────────────────────────

def export_pca_json(output_path: str, hand: str):
    """Export PCA basis + mean to JSON for Unity MANODecoder.cs.

    Unity JsonUtility cannot deserialize jagged arrays, so each PCA row is
    wrapped in a {"values": [...]} object (matches MANODecoder.FloatRow).
    """
    fk = MANOForwardKinematics(hand)
    basis, mean = fk.pca_basis()   # basis: (15, 45), mean: (45,)
    data = {
        "pca_matrix": [{"values": row.tolist()} for row in basis],  # List[{values:[45f]}]
        "mean_pose":  mean.tolist(),                                  # [45f]
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported PCA basis ({basis.shape}) → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--export_pca", metavar="PATH",
                        help="Export PCA basis JSON to PATH (for Unity)")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--test", action="store_true",
                        help="Smoke test FK with random inputs")
    args = parser.parse_args()

    if args.export_pca:
        export_pca_json(args.export_pca, args.hand)

    if args.test:
        print(f"Testing MANO FK ({args.hand})...")
        fk = MANOForwardKinematics(args.hand)
        N = 8
        pose   = np.zeros((N, 15), dtype=np.float32)
        wrist  = np.tile([1, 0, 0, 0, 1, 0], (N, 1)).astype(np.float32)  # identity
        joints = fk(pose, wrist)
        print(f"  joints shape : {joints.shape}")           # (8, 21, 3)
        print(f"  wrist offset : {joints[:, 0, :].max():.6f}  (should be ~0)")
        print(f"  finger span  : {(joints.max() - joints.min())*1000:.1f} mm")
        print("OK")
