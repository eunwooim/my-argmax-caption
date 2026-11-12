"""Synthetic DOCCI-shaped records (colored shapes drawn with PIL), used only by --mock."""
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


def image(example_id):
    a, ca, b, cb = _scene(int(example_id.split("_")[1]))
    img = Image.new("RGB", (320, 240), (128, 128, 128))
    d = ImageDraw.Draw(img)
    _draw(a, ca, (20, 60, 140, 180), d)
    _draw(b, cb, (180, 60, 300, 180), d)
    return img


def records(n=8):
    recs = []
    for i in range(1000, 1000 + n):
        a, ca, b, cb = _scene(i)
        desc = (f"A gray background with two shapes. On the left is a {ca} {a}. "
                f"On the right is a {cb} {b}. The two shapes are the same size. "
                f"Nothing else is in the picture.")
        recs.append(dict(example_id=f"synth_{i}", split="test", image_file=f"synth_{i}.jpg", description=desc))
    return recs


def caption2(example_id, version="simple", captioner="before"):
    """A fake AI caption with known mistakes, one per coverage case (i % 5):
    0 exact (positive only), 1 left color wrong (contradiction + hallucinated), 2 right shape left out
    (omission), 3 an invented cat (hallucinated only), 4 left color wrong and right left out (all three).
    i % 9 == 8: a caption of some other picture (nothing shared, so no positive: the image is dropped).
    version is kept for the call signature (only "simple" is used now).
    captioner "after" (the GRPO captioner) gets the left color right on i % 5 == 1: fewer hallucinations."""
    i = int(example_id.split("_")[1])
    if i % 9 == 8:
        return "A brown dog runs along a sandy beach under a blue sky."
    a, ca, b, cb = _scene(i)
    wrong = next(c for c in COLS if c not in (ca, cb))
    kind = i % 5
    wrong_left = kind == 4 or (kind == 1 and captioner != "after")
    left = f"On the left is a {wrong if wrong_left else ca} {a}."
    right = "" if kind in (2, 4) else f" On the right is a {cb} {b}."
    extra = " A small cat sits between them." if kind == 3 else ""
    return f"A gray background with some shapes. {left}{right}{extra}"
