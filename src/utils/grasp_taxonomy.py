"""object_name -> grasp_category. Cross-dataset ortak dil (ham obje id degil).

Kategoriler (4): hook | power | wide | pinch
- hook : kulplu, parmaklar kulpa kanca (mug, coffee_pot, spatula)
- power: silindir/dolu kavrama, tum parmak sarar (bottle, can, flask, dumbbell)
- wide : genis acikLik / kutu / tabak / kase (bowl, carton, box, plate)
- pinch: ince/kucuk, bas+isaret (marker, pen, small flat)

NOT: Atama heuristiktir, tezde bir etiketleme kararidir. Belirsizler yorumda isaretli.
v1 build filtresi 3 kategori kullanir (hook/power/wide); pinch v1.1.
Tek dogruluk kaynagi: bu dosya.
"""

# HOT3D (Quest3 + Aria) obje katalogu
HOT3D = {
    "mug_white": "hook",
    "mug_patterned": "hook",
    "coffee_pot": "hook",
    "potato_masher": "hook",     # uzun kulp
    "spatula_red": "hook",
    "bottle_bbq": "power",
    "bottle_mustard": "power",
    "bottle_ranch": "power",
    "can_parmesan": "power",
    "can_soup": "power",
    "can_tomato_sauce": "power",
    "flask": "power",
    "dumbbell_5lb": "power",
    "vase": "power",
    "bowl": "wide",
    "plate_bamboo": "wide",
    "carton_milk": "wide",
    "carton_oj": "wide",
    "birdhouse_toy": "wide",
    "puzzle_toy": "wide",
    "food_vegetables": "wide",
    "food_waffles": "wide",
    "holder_black": "wide",
    "holder_gray": "wide",
    "whiteboard_eraser": "wide",
    "keyboard": "wide",          # buyuk; pratikte v1 disi
    "whiteboard_marker": "pinch",
    "cellphone": "pinch",
    "dvd_remote": "pinch",
    "mouse": "pinch",
    "spoon_wooden": "pinch",
    "aria_small": "pinch",
    "dino_toy": "power",         # belirsiz
}

# DexYCB (YCB) -> ayni kategoriler (1C'de kullanilacak)
DEXYCB = {
    "002_master_chef_can": "power",
    "005_tomato_soup_can": "power",
    "007_tuna_fish_can": "power",
    "010_potted_meat_can": "power",
    "006_mustard_bottle": "power",
    "021_bleach_cleanser": "power",
    "019_pitcher_base": "hook",
    "025_mug": "hook",
    "003_cracker_box": "wide",
    "004_sugar_box": "wide",
    "008_pudding_box": "wide",
    "009_gelatin_box": "wide",
    "036_wood_block": "wide",
    "061_foam_brick": "wide",
    "024_bowl": "wide",
    "040_large_marker": "pinch",
    "037_scissors": "pinch",
    "011_banana": "pinch",
    "051_large_clamp": "power",
    "052_extra_large_clamp": "power",
    "035_power_drill": "hook",
}

# ARCTIC (marjinal; sadece el-ici olanlar) -> 1C opsiyonel
ARCTIC = {
    "ketchup": "power",
    "box": "wide",
    "phone": "pinch",
    "notebook": "wide",
    # buyuk eklemli cihazlar (microwave, espressomachine, capsulemachine, mixer,
    # waffleiron, laptop, scissors) v1 scope disi -> map yok
}

_ALL = {**HOT3D, **DEXYCB, **ARCTIC}

CATEGORIES = ["hook", "power", "wide", "pinch"]
CATEGORY_ID = {c: i for i, c in enumerate(CATEGORIES)}


def get_category(object_name):
    """object_name -> kategori (str) veya None (map yok / scope disi)."""
    return _ALL.get(object_name)


def category_id(object_name):
    c = get_category(object_name)
    return CATEGORY_ID[c] if c is not None else -1


if __name__ == "__main__":
    import collections
    for src, d in [("HOT3D", HOT3D), ("DexYCB", DEXYCB), ("ARCTIC", ARCTIC)]:
        cc = collections.Counter(d.values())
        print(f"{src}: {dict(cc)}  ({len(d)} obje)")
