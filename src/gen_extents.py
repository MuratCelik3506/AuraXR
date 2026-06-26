"""Faz 3 ek: canonical'da gecen TUM nesneler icin bbox extent (HOT3D GLB + DexYCB obj).

obj_size varyanti her obj_name icin extent ister; pinch eklenince eski dosya eksik kaldi.
seq npz'lerden (obj_name, obj_uid) toplar, mesh'ten extent hesaplar.
"""
import os
import json
import glob
import numpy as np
import trimesh

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(_REPO, "data", "processed", "hot3d_canonical")
HOT3D_ASSETS = os.path.join(_REPO, "data/raw/hot3d/assets/Hot3DAssets_assets_assets")
DEXYCB_MODELS = os.path.join(_REPO, "data/raw/dexycb/models")


def extent_for(name, uid, is_dx):
    if is_dx:
        p = os.path.join(DEXYCB_MODELS, name, "textured_simple.obj")
    else:
        p = os.path.join(HOT3D_ASSETS, f"{uid}.glb")
    if not os.path.exists(p):
        return None
    mesh = trimesh.load(p, force="mesh")
    return (mesh.bounds[1] - mesh.bounds[0]).tolist()


def main():
    seen = {}   # name -> (uid, is_dx)
    for npz in sorted(glob.glob(os.path.join(CANON, "seq_*.npz"))):
        sid = os.path.basename(npz)[4:-4]
        is_dx = sid.startswith("DX")
        d = np.load(npz, allow_pickle=True)
        names = d["obj_name"].astype(str)
        uids = d["obj_uid"].astype(str)
        for nm, ui in zip(names, uids):
            seen.setdefault(nm, (ui, is_dx))
    extents = {}
    missing = []
    for nm, (ui, is_dx) in sorted(seen.items()):
        e = extent_for(nm, ui, is_dx)
        if e is None:
            missing.append(nm)
        else:
            extents[nm] = e
    json.dump(extents, open(os.path.join(CANON, "object_extents.json"), "w"), indent=2)
    print(f"extents yazildi: {len(extents)} nesne | eksik mesh: {missing}")


if __name__ == "__main__":
    main()
