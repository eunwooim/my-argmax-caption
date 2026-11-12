"""Synthetic stand-ins for SugarCrepe / DOCCI / COCO train, used only by --mock.

Same item shapes as data.py, generated locally (colored shapes drawn with PIL), so the whole
pipeline runs offline. Nothing here is a benchmark.
"""
import random

from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle"]
COLS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220),
        "yellow": (240, 220, 30), "black": (10, 10, 10), "white": (250, 250, 250)}


def _scene(i):
    r = random.Random(i)
    a, b = r.sample(SHAPES, 2)
    ca, cb = r.sample(list(COLS), 2)
    return a, ca, b, cb


def _draw(shape, color, box, d):
    if shape == "circle":
        d.ellipse(box, fill=COLS[color])
    elif shape == "square":
        d.rectangle(box, fill=COLS[color])
    else:
        x0, y0, x1, y1 = box
        d.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=COLS[color])


def image(filename):
    i = int(filename.split("_")[1].split(".")[0])
    a, ca, b, cb = _scene(i)
    w, h = 320 + 16 * (i % 5), 240
    img = Image.new("RGB", (w, h), (128, 128, 128))
    d = ImageDraw.Draw(img)
    _draw(a, ca, (20, 60, 140, 180), d)
    _draw(b, cb, (w - 140, 60, w - 20, 180), d)
    return img


def sugarcrepe(n_images=40):
    items = []
    for i in range(n_images):
        a, ca, b, cb = _scene(i)
        fn = f"synth_{i}.png"
        cap = f"A {ca} {a} to the left of a {cb} {b}."
        other_col = [c for c in COLS if c not in (ca, cb)][0]
        other_shape = [s for s in SHAPES if s not in (a, b)][0]
        negs = {
            "replace_att": f"A {other_col} {a} to the left of a {cb} {b}.",
            "replace_obj": f"A {ca} {other_shape} to the left of a {cb} {b}.",
            "replace_rel": f"A {ca} {a} to the right of a {cb} {b}.",
            "swap_att": f"A {cb} {a} to the left of a {ca} {b}.",
            "swap_obj": f"A {ca} {b} to the left of a {cb} {a}.",
        }
        for t, neg in negs.items():
            items.append(dict(id=f"{t}/{i}", type=t, filename=fn, caption=cap, negative_caption=neg))
    return items


def docci(n=8):
    recs = []
    for i in range(1000, 1000 + n):
        a, ca, b, cb = _scene(i)
        desc = (f"A gray background with two shapes. On the left is a {ca} {a}. "
                f"On the right is a {cb} {b}. The two shapes are the same size. "
                f"Nothing else is in the picture.")
        recs.append(dict(example_id=f"synth_{i}", description=desc, image=image(f"synth_{i}.png")))
    return recs


def centering_sets(n=16):
    imgs = [dict(file_name=f"synth_{5000 + i}.png") for i in range(n)]
    caps = []
    for i in range(6000, 6000 + n):
        a, ca, b, cb = _scene(i)
        caps.append(f"A {ca} {a} to the left of a {cb} {b}.")
    return imgs, caps
