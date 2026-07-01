"""MANO sag-el: PCA15 -> axis-angle45 decode + (joint-only) FK.

data/models/mano_right_components.npz (convert_mano_npz.py ciktisi) okur. chumpy YOK.

Konvansiyon:
- HOT3D her el icin: pose=15 PCA (parmak), wrist_xform={t_xyz, q_wxyz} (global root pozu), betas=10.
- Global root oryantasyonu wrist_xform'dan gelir; 45 parmak axis-angle PCA decode'dan.
- FK: root identity ile model uzayinda eklemler hesaplanir, root joint orijine alinir,
  sonra (R_wrist, t_wrist) ile dunyaya tasinir. Eklem pozisyonlari icin bu *tam* (yaklasim degil).
- Mesh (viz/fingertip) icin tam LBS (posedirs + weights) saglanir.

Birim: metre. Donus: dunya cercevesi (HOT3D'nin kendi sag-elli frame'i; Unity flip Faz 3).
"""
import os
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NPZ = os.path.join(_REPO, "data", "models", "mano_right_components.npz")

# MANO sag-el fingertip vertex indeksleri (thumb, index, middle, ring, pinky)
FINGERTIP_VIDX = np.array([744, 320, 443, 555, 672], dtype=np.int64)


class ManoRight:
    def __init__(self, npz_path=_NPZ):
        d = np.load(npz_path)
        self.hands_components = d["hands_components"]      # (45,45)
        self.hands_mean = d["hands_mean"]                  # (45,)
        self.v_template = d["v_template"]                  # (778,3)
        self.shapedirs = d["shapedirs"]                    # (778,3,10)
        self.posedirs = d["posedirs"]                      # (778,3,135)
        self.J_regressor = d["J_regressor"]                # (16,778)
        self.weights = d["weights"]                        # (778,16)
        kt = d["kintree_table"].astype(np.int64)           # (2,16)
        self.parents = kt[0].copy()
        self.parents[0] = -1
        self.faces = d["f"]

    # --- PCA decode ----------------------------------------------------------
    def decode_pca(self, pca15):
        """pca (..,15) -> axis-angle (..,45). full = mean + pca @ components[:15]."""
        pca15 = np.asarray(pca15, dtype=np.float64)
        return self.hands_mean + pca15 @ self.hands_components[:15]

    # --- shape ---------------------------------------------------------------
    def shaped(self, betas):
        betas = np.asarray(betas, dtype=np.float64).reshape(-1)[:10]
        v_shaped = self.v_template + (self.shapedirs @ betas)   # (778,3)
        J = self.J_regressor @ v_shaped                          # (16,3)
        return v_shaped, J

    # --- FK ------------------------------------------------------------------
    def shaped_cached(self, betas):
        """sekans basina cache: (v_shaped, J). betas seq icinde sabit kabul."""
        key = round(float(np.asarray(betas).reshape(-1)[0]), 5)
        c = getattr(self, "_sc", {})
        if key not in c:
            c[key] = self.shaped(betas)
            self._sc = c
        return c[key]

    def fk(self, pose_aa45, betas, wrist_q_wxyz, wrist_t,
           want_mesh=False, want_fingertips=False, shaped=None):
        """Dunya cercevesinde 16 eklem (+ istege bagli 778 vertex / 5 fingertip).

        pose_aa45: (45,) parmak axis-angle. Root identity alinir.
        wrist_q_wxyz: (4,) global root quaternion. wrist_t: (3,).
        shaped: onceden hesaplanmis (v_shaped, J) -> tekrar shape hesabini atlar.
        """
        pose_aa45 = np.asarray(pose_aa45, dtype=np.float64).reshape(15, 3)
        v_shaped, J = shaped if shaped is not None else self.shaped(betas)

        full = np.zeros((16, 3), dtype=np.float64)
        full[1:] = pose_aa45
        R = _rodrigues(full)                                 # (16,3,3)

        G = np.zeros((16, 4, 4), dtype=np.float64)
        G[0] = _rt(R[0], J[0])
        for i in range(1, 16):
            G[i] = G[self.parents[i]] @ _rt(R[i], J[i] - J[self.parents[i]])
        G_packed = G.copy()
        for i in range(16):
            G_packed[i] = G[i] @ _rt(np.eye(3), -J[i])

        joints_model = G[:, :3, 3]
        R_w = _quat_to_R(wrist_q_wxyz)
        wrist_t = np.asarray(wrist_t)

        def to_world(p_model):
            return (p_model - joints_model[0]) @ R_w.T + wrist_t

        out = {"joints": to_world(joints_model)}

        if want_mesh or want_fingertips:
            pf = (R[1:] - np.eye(3)).reshape(-1)             # (135,)
            if want_mesh:
                v_posed = v_shaped + (self.posedirs @ pf)
                T = np.einsum("vj,jmn->vmn", self.weights, G_packed)
                v_h = np.concatenate([v_posed, np.ones((778, 1))], axis=1)
                verts = np.einsum("vmn,vn->vm", T, v_h)[:, :3]
                out["verts"] = to_world(verts)
                out["fingertips"] = out["verts"][FINGERTIP_VIDX]
            else:
                idx = FINGERTIP_VIDX
                v_posed_t = v_shaped[idx] + (self.posedirs[idx] @ pf)   # (5,3)
                T_t = np.einsum("vj,jmn->vmn", self.weights[idx], G_packed)  # (5,4,4)
                v_h = np.concatenate([v_posed_t, np.ones((5, 1))], axis=1)
                tips = np.einsum("vmn,vn->vm", T_t, v_h)[:, :3]
                out["fingertips"] = to_world(tips)
        return out


# --- math helpers ------------------------------------------------------------
def _rodrigues(aa):
    """(N,3) axis-angle -> (N,3,3) rotation."""
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(aa, axis=1, keepdims=True)        # (N,1)
    out = np.tile(np.eye(3), (aa.shape[0], 1, 1))
    nz = theta[:, 0] > 1e-8
    if np.any(nz):
        k = aa[nz] / theta[nz]
        th = theta[nz, 0]
        K = np.zeros((k.shape[0], 3, 3))
        K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
        I = np.eye(3)[None]
        s = np.sin(th)[:, None, None]; c = np.cos(th)[:, None, None]
        out[nz] = I + s * K + (1 - c) * (K @ K)
    return out


def _rt(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t).reshape(3)
    return M


def _quat_to_R(q_wxyz):
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


if __name__ == "__main__":
    # tek-frame smoke-test: bir HOT3D frame'i decode + FK
    import json, zipfile, glob
    m = ManoRight()
    seq_dir = glob.glob(os.path.join(_REPO, "data/raw/hot3d/quest3/train/*"))[0]
    hz = glob.glob(os.path.join(seq_dir, "*hand_data.zip"))[0]
    with zipfile.ZipFile(hz) as z:
        with z.open("mano_hand_pose_trajectory.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                hp = rec.get("hand_poses", {})
                if hp:
                    break
    print("seq:", os.path.basename(seq_dir), "| hand ids:", list(hp.keys()))
    for hid, h in hp.items():
        aa45 = m.decode_pca(h["pose"])
        wx = h["wrist_xform"]
        res = m.fk(aa45, h["betas"], wx["q_wxyz"], wx["t_xyz"], want_mesh=True)
        J = res["joints"]; tips = res["fingertips"]
        print(f"  hand {hid}: joints {J.shape} finite={np.isfinite(J).all()}")
        print(f"    wrist_t={np.round(wx['t_xyz'],3)} | joint0={np.round(J[0],3)}")
        print(f"    hand span (m)={np.round(np.linalg.norm(J.max(0)-J.min(0)),3)} "
              f"| fingertip span={np.round(np.linalg.norm(tips.max(0)-tips.min(0)),3)}")
