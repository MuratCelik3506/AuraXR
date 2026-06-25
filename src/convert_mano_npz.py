"""Tek seferlik: MANO_RIGHT.pkl -> saf-numpy data/models/mano_right_components.npz

MANO .pkl chumpy-tabanlidir. chumpy py3.13'te build olmadigindan, burada chumpy'yi
sahte (shim) bir modulle degistirip pickle'i okuyoruz. Cikti tamamen numpy; sonraki
tum calistirmalar (mano_right.py vb.) chumpy'siz bu npz'i okur.

Kullanim:
    .venv/bin/python src/convert_mano_npz.py
"""
import os
import sys
import types
import pickle

import numpy as np
import scipy.sparse


# --- chumpy shim --------------------------------------------------------------
# chumpy Ch nesneleri normal python sinifi gibi picklelenir:
#   cls.__new__(cls) ; obj.__setstate__(state)  (state genelde __dict__)
# Leaf Ch'in sayisal degeri __dict__['x'] icindedir. __array__ ile numpy'e cevrilir.
class _ChShim:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, tuple):
            # (args, dict) bicimi
            state = next((s for s in state if isinstance(s, dict)), {})
        if isinstance(state, dict):
            self.__dict__.update(state)

    def _resolve(self):
        d = self.__dict__
        # leaf
        for k in ("x", "_result", "r"):
            if k in d and d[k] is not None:
                return np.asarray(d[k])
        # reorder/reshape dugumu: child 'a' -> ravel()[idxs] -> reshape(preferred_shape)
        if "a" in d and d["a"] is not None:
            child = d["a"]
            cv = child._resolve() if isinstance(child, _ChShim) else np.asarray(child)
            if "idxs" in d and d["idxs"] is not None:
                idxs = np.asarray(d["idxs"]).astype(np.int64)
                res = cv.ravel()[idxs]
            else:
                res = cv
            ps = d.get("preferred_shape")
            if ps is not None:
                res = res.reshape(tuple(int(x) for x in ps))
            return res
        return np.asarray([])

    def __array__(self, dtype=None):
        v = self._resolve()
        return np.asarray(v, dtype=dtype) if dtype is not None else v


def _install_chumpy_shim():
    mod = types.ModuleType("chumpy")
    sub = types.ModuleType("chumpy.ch")
    for m in (mod, sub):
        # bilinen tum chumpy sinif isimlerini ayni shim'e bagla
        for name in ("Ch", "ChHandle", "reordering", "Select", "Concatenate"):
            setattr(m, name, _ChShim)
    mod.ch = sub
    sys.modules["chumpy"] = mod
    sys.modules["chumpy.ch"] = sub


class _ManoUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChShim
        return super().find_class(module, name)


def _to_dense(v):
    """chumpy/scipy/ndarray -> dense float numpy."""
    if scipy.sparse.issparse(v):
        return np.asarray(v.todense(), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_pkl = os.path.join(repo, "MANO_RIGHT.pkl")
    out_dir = os.path.join(repo, "data", "models")
    out_npz = os.path.join(out_dir, "mano_right_components.npz")
    os.makedirs(out_dir, exist_ok=True)

    _install_chumpy_shim()
    with open(src_pkl, "rb") as f:
        d = _ManoUnpickler(f, encoding="latin1").load()

    print("pkl keys:", sorted(d.keys()))

    # Cikarilacak alanlar (joint-only FK + viz + PCA decode icin yeterli)
    out = {}

    # PCA decode
    out["hands_components"] = _to_dense(d["hands_components"])  # (ncomp, 45)
    out["hands_mean"] = _to_dense(d["hands_mean"]).reshape(-1)  # (45,)
    if "hands_coeffs" in d:
        out["hands_coeffs"] = _to_dense(d["hands_coeffs"])

    # FK / sekil
    out["v_template"] = _to_dense(d["v_template"]).reshape(-1, 3)        # (778,3)
    out["shapedirs"] = _to_dense(d["shapedirs"])                          # (778,3,10)
    out["posedirs"] = _to_dense(d["posedirs"])                            # (778,3,135)
    out["J_regressor"] = _to_dense(d["J_regressor"])                      # (16,778)
    out["weights"] = _to_dense(d["weights"])                             # (778,16)
    out["kintree_table"] = np.asarray(d["kintree_table"]).astype(np.int64)  # (2,16)
    out["f"] = np.asarray(d["f"]).astype(np.int64)                       # (1538,3)

    # Sanity
    for k, v in out.items():
        if np.asarray(v).size == 0:
            raise RuntimeError(f"BOS alan (shim cozemedi): {k}")
        print(f"  {k}: {np.asarray(v).shape} {np.asarray(v).dtype}")

    np.savez(out_npz, **out)
    print(f"\nyazildi -> {out_npz}")


if __name__ == "__main__":
    main()
