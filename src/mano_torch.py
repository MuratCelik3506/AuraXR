"""Differentiable (torch) MANO sag-el: PCA15 -> joint-only FK.

Egitimde FK pozisyon loss (MPJPE) icin. mano_right.py (numpy) ile parite hedefli.
Joint-only: J(betas) = J_base + J_dirs @ betas (778 vertex'e gerek yok).
"""
import os
import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NPZ = os.path.join(_REPO, "data", "models", "mano_right_components.npz")


class ManoTorchFK(nn.Module):
    def __init__(self, npz_path=_NPZ):
        super().__init__()
        d = np.load(npz_path)
        comp = torch.tensor(d["hands_components"][:15], dtype=torch.float32)  # (15,45)
        mean = torch.tensor(d["hands_mean"], dtype=torch.float32)             # (45,)
        Jreg = d["J_regressor"]                                               # (16,778)
        vtemp = d["v_template"]                                               # (778,3)
        sdirs = d["shapedirs"]                                                # (778,3,10)
        J_base = Jreg @ vtemp                                                 # (16,3)
        J_dirs = np.einsum("jv,vdb->jdb", Jreg, sdirs)                        # (16,3,10)
        kt = d["kintree_table"].astype(np.int64)
        parents = kt[0].copy(); parents[0] = -1

        self.register_buffer("comp", comp)
        self.register_buffer("mean", mean)
        self.register_buffer("J_base", torch.tensor(J_base, dtype=torch.float32))
        self.register_buffer("J_dirs", torch.tensor(J_dirs, dtype=torch.float32))
        self.parents = parents.tolist()

    def decode(self, pca15):
        """(N,15) -> (N,45) axis-angle."""
        return self.mean + pca15 @ self.comp

    def joints(self, pca15, betas, wrist_q, wrist_t):
        """(N,15),(N,10),(N,4 wxyz),(N,3) -> (N,16,3) dunya eklemleri."""
        N = pca15.shape[0]
        dev = pca15.device
        aa45 = self.decode(pca15).reshape(N, 15, 3)
        full = torch.zeros(N, 16, 3, device=dev, dtype=pca15.dtype)
        full[:, 1:] = aa45
        R = _rodrigues(full.reshape(N * 16, 3)).reshape(N, 16, 3, 3)

        J = self.J_base + torch.einsum("jdb,nb->njd", self.J_dirs, betas)     # (N,16,3)

        # zincir: G_i (N,4,4)
        G = [None] * 16
        G[0] = _rt(R[:, 0], J[:, 0])
        for i in range(1, 16):
            p = self.parents[i]
            G[i] = G[p] @ _rt(R[:, i], J[:, i] - J[:, p])
        joints_model = torch.stack([G[i][:, :3, 3] for i in range(16)], dim=1)  # (N,16,3)

        Rw = _quat_to_R(wrist_q)                                              # (N,3,3)
        local = joints_model - joints_model[:, :1]
        world = torch.einsum("nij,nkj->nki", Rw, local) + wrist_t[:, None]
        return world


def _rodrigues(aa):
    """(N,3) -> (N,3,3) differentiable."""
    N = aa.shape[0]
    theta = aa.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k = aa / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    z = torch.zeros(N, device=aa.device, dtype=aa.dtype)
    K = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], dim=1).reshape(N, 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(N, 3, 3)
    s = torch.sin(theta)[:, :, None]
    c = torch.cos(theta)[:, :, None]
    return I + s * K + (1 - c) * (K @ K)


def _rt(R, t):
    """R(N,3,3) t(N,3) -> (N,4,4)."""
    N = R.shape[0]
    M = torch.zeros(N, 4, 4, device=R.device, dtype=R.dtype)
    M[:, :3, :3] = R
    M[:, :3, 3] = t
    M[:, 3, 3] = 1.0
    return M


def _quat_to_R(q):
    """(N,4 wxyz) -> (N,3,3)."""
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    return R


if __name__ == "__main__":
    # numpy FK ile parite
    from mano_right import ManoRight
    mr = ManoRight()
    mt = ManoTorchFK()
    rng = np.random.RandomState(0)
    pca = rng.randn(5, 15).astype(np.float32) * 0.5
    betas = rng.randn(5, 10).astype(np.float32)
    q = rng.randn(5, 4).astype(np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    t = rng.randn(5, 3).astype(np.float32)
    jt = mt.joints(torch.tensor(pca), torch.tensor(betas), torch.tensor(q), torch.tensor(t)).detach().numpy()
    err = 0
    for i in range(5):
        aa = mr.decode_pca(pca[i])
        jn = mr.fk(aa, betas[i], q[i], t[i])["joints"]
        err = max(err, np.abs(jt[i] - jn).max())
    print(f"torch-FK vs numpy-FK max abs err = {err:.2e}  ({'OK' if err < 1e-4 else 'FARK'})")
