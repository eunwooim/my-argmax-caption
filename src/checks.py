"""Checks 1-4 (doc section 5), bootstrap intervals, and the GRPO gate (doc section 6)."""
import random

import numpy as np
import torch

import data as D


# ------------------------------------------------------------------ bootstrap
def bootstrap(outcomes, groups, B=10000, seed=0, chunk=500):
    """Mean of 0/1 outcomes with a 95% interval from B resamples over groups (images)."""
    outcomes = np.asarray(outcomes, dtype=float)
    if len(outcomes) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, n_images=0)
    _, gidx = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    G = gidx.max() + 1
    s = np.bincount(gidx, weights=outcomes, minlength=G)
    c = np.bincount(gidx, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    stats = []
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        idx = rng.integers(0, G, size=(b, G))
        stats.append(s[idx].sum(1) / c[idx].sum(1))
    stats = np.concatenate(stats)
    return dict(mean=float(outcomes.mean()), lo=float(np.percentile(stats, 2.5)),
                hi=float(np.percentile(stats, 97.5)), n=int(len(outcomes)), n_images=int(G))


# ------------------------------------------------------------------ similarity
class Scorer:
    """cos(img, cap) for one variant = (score, centered, layer), reading from an embedding cache."""

    def __init__(self, cache, layer, means=None):
        self.cache, self.li = cache, layer - 1
        self.mu_i = self.mu_c = None
        if means is not None:
            self.mu_i, self.mu_c = means["image"][self.li], means["caption"][self.li]

    def vec(self, key, kind):
        v = self.cache.vec(key, self.li)
        mu = self.mu_i if kind == "image" else self.mu_c
        return v - mu if mu is not None else v

    def __call__(self, img_key, caption):
        return float(torch.nn.functional.cosine_similarity(
            self.vec(img_key, "image"), self.vec(("cap", caption), "caption"), dim=0))


# ------------------------------------------------------------------ what each check needs embedded
def sugarcrepe_keys(items):
    imgs = {("img", it["filename"]) for it in items}
    caps = {c for it in items for c in (it["caption"], it["negative_caption"])}
    return imgs, caps


def docci_plan(docci):
    """Per record: prefix captions (check 2) and true/swapped full description (check 3)."""
    plan = []
    for j, r in enumerate(docci):
        sents = D.split_sentences(r["description"])
        prefixes = [" ".join(sents[:k]) for k in range(1, len(sents) + 1)]
        sw = D.swap_one_color(sents, seed=j)
        swapped = " ".join(sw[1]) if sw else None
        plan.append(dict(id=r["example_id"], prefixes=prefixes, true=" ".join(sents), swapped=swapped))
    return plan


def docci_keys(plan):
    imgs = {("docci", p["id"]) for p in plan}
    caps = {c for p in plan for c in p["prefixes"] + [p["true"]] + ([p["swapped"]] if p["swapped"] else [])}
    return imgs, caps


def noise_plan(test_items, seed=0):
    """One true caption per held-out image, plus a different random held-out image."""
    first = {}
    for it in sorted(test_items, key=lambda x: x["id"]):
        first.setdefault(it["filename"], it["caption"])
    fns = sorted(first)
    rng = random.Random(seed)
    plan = []
    for i, fn in enumerate(fns):
        j = rng.randrange(len(fns) - 1)
        other = fns[j if j < i else j + 1]
        plan.append(dict(filename=fn, caption=first[fn], wrong=other, noise_seed=i))
    return plan


def noise_keys(plan):
    imgs = {("img", p["filename"]) for p in plan} | {("img", p["wrong"]) for p in plan} \
        | {("noise", p["filename"], p["noise_seed"]) for p in plan}
    caps = {p["caption"] for p in plan}
    return imgs, caps


# ------------------------------------------------------------------ the four checks
def check1_sugarcrepe(sim, items, B):
    out = {}
    rows = [(it["type"], it["filename"],
             sim(("img", it["filename"]), it["caption"]) > sim(("img", it["filename"]), it["negative_caption"]))
            for it in items]
    for t in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == t]
        out[t] = bootstrap([r[2] for r in sub], [r[1] for r in sub], B)
    out["overall"] = bootstrap([r[2] for r in rows], [r[1] for r in rows], B)
    return out


def check2_docci_more(sim, plan, B):
    steps, groups, curves = [], [], []
    for p in plan:
        s = [sim(("docci", p["id"]), c) for c in p["prefixes"]]
        curves.append(s)
        for k in range(1, len(s)):
            steps.append(s[k] > s[k - 1])
            groups.append(p["id"])
    K = max(len(c) for c in curves)
    curve = [dict(k=k, mean=float(np.mean([c[k - 1] for c in curves if len(c) >= k])),
                  n=sum(len(c) >= k for c in curves)) for k in range(1, K + 1)]
    return bootstrap(steps, groups, B), curve


def check3_docci_wrong(sim, plan, B):
    rows = [(p["id"], sim(("docci", p["id"]), p["true"]) > sim(("docci", p["id"]), p["swapped"]))
            for p in plan if p["swapped"]]
    return bootstrap([r[1] for r in rows], [r[0] for r in rows], B)


def check4_noise(sim, plan, B):
    vs_noise, vs_wrong, both, g = [], [], [], []
    for p in plan:
        real = sim(("img", p["filename"]), p["caption"])
        noise = sim(("noise", p["filename"], p["noise_seed"]), p["caption"])
        wrong = sim(("img", p["wrong"]), p["caption"])
        vs_noise.append(real > noise)
        vs_wrong.append(real > wrong)
        both.append(real > noise and real > wrong)
        g.append(p["filename"])
    return dict(vs_noise=bootstrap(vs_noise, g, B), vs_wrong=bootstrap(vs_wrong, g, B),
                vs_both=bootstrap(both, g, B))


# ------------------------------------------------------------------ gate
def gate(res, threshold=0.60):
    """Doc 6: all on held-out data."""
    sc, dw = res["check1"]["overall"], res["check3"]
    nz, wr = res["check4"]["vs_noise"], res["check4"]["vs_wrong"]
    conds = {
        f"SugarCrepe >= {threshold:.0%} and lower end > 50%": sc["mean"] >= threshold and sc["lo"] > 0.5,
        f"DOCCI wrong-detail >= {threshold:.0%} and lower end > 50%": dw["mean"] >= threshold and dw["lo"] > 0.5,
        "Real image beats noise, lower end > 50%": nz["lo"] > 0.5,
        "Real image beats wrong image, lower end > 50%": wr["lo"] > 0.5,
    }
    return dict(passed=all(conds.values()), conditions=conds)
