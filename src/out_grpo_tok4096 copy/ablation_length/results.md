# Ablation: is the reward gain just shorter captions?

Reward **S2-centered@L24** (the GRPO training reward; LoRA off), source `../argmax_caption/out_grpo_tok4096`. Each row scores a different caption of the same images. *before* / *after* are grpo.py's base and GRPO captions; the *before_* rows are the base caption cut shorter, with no model change; *short_* rows are the base model asked for a short caption. Differences are per image, 95% bootstrap over images.

If length alone explained the gain, the shortened rows would reach *after*. They don't if `after − row` stays above 0.

## coco_val (n=500)

| caption | words | score | row − before | after − row | after beats row |
|---|---|---|---|---|---|
| before | 99.7 | 0.7007 | +0.0000 [+0.0000, +0.0000] | +0.0412 [+0.0372, +0.0452] | 85.4% |
| after | 65.2 | 0.7420 | +0.0412 [+0.0372, +0.0452] | — | — |
| before_lenmatch | 73.4 | 0.7186 | +0.0178 [+0.0147, +0.0209] | +0.0234 [+0.0200, +0.0269] | 73.4% |
| before_words | 65.1 | 0.6729 | -0.0278 [-0.0350, -0.0208] | +0.0690 [+0.0620, +0.0765] | 84.6% |
| before_k1 | 15.8 | 0.6543 | -0.0464 [-0.0565, -0.0364] | +0.0876 [+0.0785, +0.0969] | 81.4% |
| before_k2 | 34.9 | 0.6932 | -0.0076 [-0.0144, -0.0007] | +0.0488 [+0.0427, +0.0550] | 76.0% |
| before_k3 | 53.3 | 0.7120 | +0.0113 [+0.0066, +0.0160] | +0.0299 [+0.0254, +0.0347] | 70.0% |
| short_1 (`Describe this image in one sentence.`) | 24.0 | 0.7324 | +0.0316 [+0.0264, +0.0367] | +0.0096 [+0.0053, +0.0139] | 56.2% |
| short_2 (`Briefly describe this image.`) | 82.8 | 0.7139 | +0.0131 [+0.0097, +0.0165] | +0.0281 [+0.0249, +0.0315] | 81.2% |

Within the *before* captions, Pearson r(words, score) = -0.081 (negative would mean shorter base captions already score higher).
Re-scored *before* vs grpo.py's stored reward_before: max |difference| 0.00e+00.

**Verdict (coco_val):** the best non-RL caption is `short_1` (0.7324 vs after 0.7420). After stays significantly above every shortened caption, so shortness alone does not explain the gain.

## docci_ood (n=200)

| caption | words | score | row − before | after − row | after beats row |
|---|---|---|---|---|---|
| before | 96.9 | 0.6740 | +0.0000 [+0.0000, +0.0000] | +0.0380 [+0.0319, +0.0445] | 86.5% |
| after | 68.3 | 0.7121 | +0.0380 [+0.0319, +0.0445] | — | — |
| before_lenmatch | 76.5 | 0.6884 | +0.0144 [+0.0098, +0.0190] | +0.0237 [+0.0174, +0.0305] | 72.0% |
| before_words | 68.2 | 0.6434 | -0.0306 [-0.0427, -0.0195] | +0.0686 [+0.0562, +0.0819] | 80.0% |
| before_k1 | 15.5 | 0.6483 | -0.0257 [-0.0398, -0.0128] | +0.0638 [+0.0502, +0.0784] | 73.5% |
| before_k2 | 33.9 | 0.6771 | +0.0031 [-0.0074, +0.0126] | +0.0350 [+0.0249, +0.0461] | 63.0% |
| before_k3 | 52.6 | 0.6916 | +0.0176 [+0.0097, +0.0248] | +0.0205 [+0.0124, +0.0293] | 60.0% |
| short_1 (`Describe this image in one sentence.`) | 21.5 | 0.7179 | +0.0438 [+0.0356, +0.0523] | -0.0058 [-0.0119, +0.0006] | 43.5% |
| short_2 (`Briefly describe this image.`) | 79.7 | 0.6845 | +0.0105 [+0.0048, +0.0159] | +0.0276 [+0.0222, +0.0330] | 79.5% |

Within the *before* captions, Pearson r(words, score) = -0.034 (negative would mean shorter base captions already score higher).
Re-scored *before* vs grpo.py's stored reward_before: max |difference| 0.00e+00.

**Verdict (docci_ood):** the best non-RL caption is `short_1` (0.7179 vs after 0.7121). After is **not** significantly above it, so length may explain part or all of the gain.

