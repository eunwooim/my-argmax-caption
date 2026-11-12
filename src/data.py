"""Data loading for Step 1 (doc section 4): SugarCrepe, DOCCI, COCO train2017 (for centering).

All downloads are cached under --cache_dir. COCO is read from a local copy when one is found
(COCO_ROOT, default /mnt/data2/shared_dataset/COCO-2017); otherwise images are fetched over http (the
doc notes https failed certificate checks in an earlier run).
"""
import glob
import io
import json
import os
import random
import re
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image

SUGARCREPE_TYPES = ["replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]
SUGARCREPE_URL = "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/{}.json"
COCO_VAL_URL = "http://images.cocodataset.org/val2017/{}"
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DOCCI_JSONL = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"
DOCCI_IMAGES_TAR = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"

# Fixed color list for the DOCCI wrong-detail check (doc 5.3, "our choice").
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink",
          "brown", "black", "white", "gray"]


# ------------------------------------------------------------------ local COCO
COCO_ROOT = os.environ.get("COCO_ROOT", "/mnt/data2/shared_dataset/COCO-2017")
_local = {}


def _find_local(kind, split):
    """Local COCO path, or None. kind 'ann' -> captions_<split>.json, kind 'img' -> folder of <split> jpgs.

    Looks a few levels deep under COCO_ROOT, so <root>/train2017, <root>/images/train2017,
    <root>/annotations/captions_train2017.json and similar layouts all work.
    """
    key = (kind, split)
    if key not in _local:
        found = None
        if os.path.isdir(COCO_ROOT):
            for depth in range(4):
                pat = os.path.join(COCO_ROOT, *(["*"] * depth),
                                   f"captions_{split}.json" if kind == "ann" else split)
                hits = sorted(p for p in glob.glob(pat)
                              if (os.path.isfile(p) if kind == "ann" else os.path.isdir(p)))
                if kind == "img":  # the folder that holds the jpgs (some copies nest train2017/train2017)
                    hits = [h for h in hits if glob.glob(os.path.join(h, "*.jpg"))[:1]]
                if hits:
                    found = hits[0]
                    break
        _local[key] = found
        print(f"[coco] {split} {'captions' if kind == 'ann' else 'images'}: "
              f"{found or 'not found under ' + COCO_ROOT + ', downloading'}", flush=True)
    return _local[key]


def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        if not url.lower().endswith((".jpg", ".png")):  # big files: say what is happening
            print(f"[download] {url}", flush=True)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def load_image(path):
    return Image.open(path).convert("RGB")


# ------------------------------------------------------------------ SugarCrepe
def load_sugarcrepe(cache_dir, types=SUGARCREPE_TYPES):
    """Items: dict(id, type, filename, caption, negative_caption)."""
    items = []
    for t in types:
        p = _download(SUGARCREPE_URL.format(t), os.path.join(cache_dir, "sugarcrepe", f"{t}.json"))
        with open(p) as f:
            d = json.load(f)
        for k, v in d.items():
            items.append(dict(id=f"{t}/{k}", type=t, filename=v["filename"],
                              caption=v["caption"], negative_caption=v["negative_caption"]))
    return items


def split_by_image(items, dev_frac=0.1, seed=0):
    """Split items by image: dev_frac of images go to dev, the rest to test."""
    images = sorted({it["filename"] for it in items})
    perm = np.random.default_rng(seed).permutation(len(images))
    n_dev = int(round(dev_frac * len(images)))
    dev_imgs = {images[i] for i in perm[:n_dev]}
    dev = [it for it in items if it["filename"] in dev_imgs]
    test = [it for it in items if it["filename"] not in dev_imgs]
    return dev, test


def coco_val_image(filename, cache_dir):
    local = _find_local("img", "val2017")
    if local and os.path.exists(os.path.join(local, filename)):
        return load_image(os.path.join(local, filename))
    p = os.path.join(cache_dir, "coco_val2017", filename)
    _download(COCO_VAL_URL.format(filename), p)
    return load_image(p)


# ------------------------------------------------------------------ COCO train2017 (centering)
def load_coco_captions(cache_dir, split="train2017", ann_path=None):
    """Returns (images: list of dict(id, file_name, coco_url), captions_by_image: dict id -> [str])."""
    if ann_path is None:
        ann_path = _find_local("ann", split)
    if ann_path is None:
        ann_path = os.path.join(cache_dir, "coco_ann", f"captions_{split}.json")
        if not os.path.exists(ann_path):
            z = _download(COCO_ANN_ZIP, os.path.join(cache_dir, "coco_ann", "annotations_trainval2017.zip"))
            with zipfile.ZipFile(z) as zf:
                with zf.open(f"annotations/captions_{split}.json") as src, open(ann_path, "wb") as dst:
                    dst.write(src.read())
    with open(ann_path) as f:
        ann = json.load(f)
    caps = {}
    for a in ann["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"].strip())
    images = sorted(ann["images"], key=lambda x: x["id"])
    return images, caps


def load_coco_train_captions(cache_dir, ann_path=None):
    return load_coco_captions(cache_dir, "train2017", ann_path)


def coco_image(img_info, cache_dir, split="train2017"):
    local = _find_local("img", split)
    if local and os.path.exists(os.path.join(local, img_info["file_name"])):
        return load_image(os.path.join(local, img_info["file_name"]))
    p = os.path.join(cache_dir, f"coco_{split}", img_info["file_name"])
    url = img_info["coco_url"].replace("https://", "http://")
    _download(url, p)
    return load_image(p)


def coco_train_image(img_info, cache_dir):
    return coco_image(img_info, cache_dir, "train2017")


def centering_sets(cache_dir, n=10000, seed=0, ann_path=None):
    """n images for the image average, and one caption from each of n *other* images."""
    images, caps = load_coco_train_captions(cache_dir, ann_path)
    images = [im for im in images if im["id"] in caps]
    rng = random.Random(seed)
    picked = rng.sample(range(len(images)), 2 * n)
    img_set = [images[i] for i in picked[:n]]
    cap_set = [rng.choice(caps[images[i]["id"]]) for i in picked[n:]]
    return img_set, cap_set


# ------------------------------------------------------------------ DOCCI
def _docci_sample(records, n, seed):
    test = sorted([r for r in records if r["split"] == "test"], key=lambda r: r["example_id"])
    idx = random.Random(seed).sample(range(len(test)), n)
    return [test[i] for i in idx]


def load_docci(cache_dir, n=200, seed=0, docci_dir=None):
    """Returns list of dict(example_id, description, image: PIL). Test split, n images, seed 0.

    Sources, in order: --docci_dir (docci_descriptions.jsonlines + images/), the official
    GCS release (descriptions jsonl + streaming only the needed files out of the image tar).
    """
    out_dir = os.path.join(cache_dir, "docci")
    if docci_dir:
        with open(os.path.join(docci_dir, "docci_descriptions.jsonlines")) as f:
            recs = _docci_sample([json.loads(l) for l in f], n, seed)
        for r in recs:
            r["image"] = load_image(os.path.join(docci_dir, "images", r["image_file"]))
        return recs

    jl = _download(DOCCI_JSONL, os.path.join(out_dir, "docci_descriptions.jsonlines"))
    with open(jl) as f:
        recs = _docci_sample([json.loads(l) for l in f], n, seed)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    missing = {r["image_file"] for r in recs if not os.path.exists(os.path.join(img_dir, r["image_file"]))}
    if missing:
        print(f"[docci] streaming {len(missing)} images out of the 3.8 GB image tar (one pass) ...")
        with urllib.request.urlopen(DOCCI_IMAGES_TAR) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for m in tf:
                name = os.path.basename(m.name)
                if m.isfile() and name in missing:
                    data = tf.extractfile(m).read()
                    with open(os.path.join(img_dir, name), "wb") as g:
                        g.write(data)
                    missing.discard(name)
                    if not missing:
                        break
    for r in recs:
        r["image"] = load_image(os.path.join(img_dir, r["image_file"]))
    return recs


def split_sentences(description):
    """Doc 4: split each description into sentences at '. '."""
    parts = [p.strip() for p in description.strip().split(". ") if p.strip()]
    return [p if p.endswith(".") else p + "." for p in parts]


_COLOR_RE = re.compile(r"\b(" + "|".join(COLORS + ["grey"]) + r")\b", re.IGNORECASE)


def swap_one_color(sentences, seed):
    """Swap one color word in one sentence for a different color from COLORS.

    Picks the first sentence that contains a color word, its first color word, and a
    different color drawn with a per-image seed. Returns (index, new_sentences) or None.
    """
    rng = random.Random(seed)
    for i, s in enumerate(sentences):
        m = _COLOR_RE.search(s)
        if not m:
            continue
        old = m.group(1).lower()
        old = "gray" if old == "grey" else old
        new = rng.choice([c for c in COLORS if c != old])
        if m.group(1)[0].isupper():
            new = new.capitalize()
        out = list(sentences)
        out[i] = s[:m.start()] + new + s[m.end():]
        return i, out
    return None


# ------------------------------------------------------------------ noise image
def noise_image(size, seed):
    w, h = size
    arr = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")
