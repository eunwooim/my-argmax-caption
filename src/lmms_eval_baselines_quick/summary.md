# lmms-eval: selfval_bon, selfval_fta, logiccheck

Plan frozen 2026-11-06 23:49:22; lmms-eval 0.7.4; model args `min_pixels=3136,max_pixels=802816`; seed 0; **limit 4 per task (smoke test)**.
Cells: value (difference to selfval_bon). ↑ / ↓ = higher / lower is better.

Not run yet: selfval_bon/mcq, selfval_bon/general, selfval_fta/mcq, selfval_fta/general

Prompting baselines (base model + another paper's method; settings and deviations in BASELINES.md): selfval_bon = selfval_bon, selfval_fta = selfval_fta, logiccheck = logiccheck

## mcq (mmstar)

| task | metric | selfval_bon | selfval_fta | logiccheck |
|---|---|---|---|---|
| mmstar | average ↑ |  |  | 0.0000 |
| mmstar | coarse perception ↑ |  |  | 0.0000 |

## caption (coco_cap_chair, docci_caption), gen_kwargs `max_new_tokens=128,temperature=0`

| task | metric | selfval_bon | selfval_fta | logiccheck |
|---|---|---|---|---|
| coco_cap_chair | coco_cap_chair_i ↓ | 8.6957 | 0.0000 (-8.6957) | 8.3333 (-0.3623) |
| coco_cap_chair | coco_cap_chair_s ↓ | 25.00 | 0.0000 (-25.0000) | 25.00 (+0.0000) |
| coco_cap_chair | coco_cap_recall ↑ | 83.33 | 83.33 (+0.0000) | 83.33 (+0.0000) |
| coco_cap_chair | words per caption | 107.25 | 98.75 (-8.5000) | 95.50 (-11.7500) |
| docci_caption | docci_Bleu_4 ↑ | 0.0677 | 0.0921 (+0.0244) | 0.0578 (-0.0099) |
| docci_caption | docci_CAPTURE ↑ | 0.5686 | 0.5920 (+0.0234) | 0.5553 (-0.0134) |
| docci_caption | docci_CIDEr ↑ | 0.0164 | 0.0046 (-0.0118) | 0.1960 (+0.1797) |
| docci_caption | docci_METEOR ↑ | 0.1256 | 0.1577 (+0.0322) | 0.1300 (+0.0044) |
| docci_caption | docci_ROUGE_L ↑ | 0.2412 | 0.2695 (+0.0283) | 0.2479 (+0.0067) |
| docci_caption | words per caption | 95.25 | 110.00 (+14.7500) | 84.75 (-10.5000) |

## general (mmbench_en_dev_static, mme, mmvet), gen_kwargs `temperature=0`

| task | metric | selfval_bon | selfval_fta | logiccheck |
|---|---|---|---|---|
| mmbench_en_dev_static | gpt_eval_score ↑ |  |  | 100.00 |
| mme | mme_cognition_score ↑ |  |  | 50.00 |
| mmvet | gpt_eval_score ↑ |  |  | 100.00 |

Result files:

- selfval_bon/caption: `lmms_eval_baselines_quick/selfval_bon/caption/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261108_002013_results.json`
- selfval_fta/caption: `lmms_eval_baselines_quick/selfval_fta/caption/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261108_002503_results.json`
- logiccheck/mcq: `lmms_eval_baselines_quick/logiccheck/mcq/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261108_001936_results.json`
- logiccheck/caption: `lmms_eval_baselines_quick/logiccheck/caption/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261108_005017_results.json`
- logiccheck/general: `lmms_eval_baselines_quick/logiccheck/general/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261108_010247_results.json`
