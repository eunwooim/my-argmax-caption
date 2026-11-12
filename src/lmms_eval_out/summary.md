# lmms-eval: base, self, external, length

Plan frozen 2026-11-03 10:30:22; lmms-eval 0.7.4; model args `min_pixels=3136,max_pixels=802816`; seed 0.
Cells: value (difference to base). ↑ / ↓ = higher / lower is better.

## mcq (pope, mmstar)

| task | metric | base | self | external | length |
|---|---|---|---|---|---|
| mmstar | average ↑ | 0.6093 | 0.6127 (+0.0033) | 0.6113 (+0.0020) | 0.6107 (+0.0013) |
| mmstar | coarse perception ↑ | 0.7120 | 0.7120 (+0.0000) | 0.7120 (+0.0000) | 0.7080 (-0.0040) |
| mmstar | fine-grained perception ↑ | 0.5160 | 0.5320 (+0.0160) | 0.5160 (+0.0000) | 0.5160 (+0.0000) |
| mmstar | instance reasoning ↑ | 0.6840 | 0.6920 (+0.0080) | 0.6920 (+0.0080) | 0.6920 (+0.0080) |
| mmstar | logical reasoning ↑ | 0.6560 | 0.6600 (+0.0040) | 0.6600 (+0.0040) | 0.6560 (+0.0000) |
| mmstar | math ↑ | 0.6520 | 0.6480 (-0.0040) | 0.6480 (-0.0040) | 0.6520 (+0.0000) |
| mmstar | science & technology ↑ | 0.4360 | 0.4320 (-0.0040) | 0.4400 (+0.0040) | 0.4400 (+0.0040) |
| pope | pope_accuracy ↑ | 0.8777 | 0.8776 (-0.0001) | 0.8777 (+0.0000) | 0.8777 (+0.0000) |
| pope | pope_f1_score ↑ | 0.8652 | 0.8651 (-0.0001) | 0.8654 (+0.0001) | 0.8652 (-0.0000) |
| pope | pope_precision ↑ | 0.9632 | 0.9629 (-0.0003) | 0.9622 (-0.0010) | 0.9635 (+0.0003) |
| pope | pope_recall ↑ | 0.7853 | 0.7853 (+0.0000) | 0.7862 (+0.0009) | 0.7851 (-0.0002) |
| pope | pope_yes_ratio ↑ | 0.4077 | 0.4078 (+0.0001) | 0.4086 (+0.0009) | 0.4074 (-0.0002) |

## caption (coco_cap_chair, docci_caption), gen_kwargs `max_new_tokens=128,temperature=0`

| task | metric | base | self | external | length |
|---|---|---|---|---|---|
| coco_cap_chair | coco_cap_chair_i ↓ | 8.5106 | 8.4998 (-0.0108) | 7.9523 (-0.5584) | 7.6994 (-0.8112) |
| coco_cap_chair | coco_cap_chair_s ↓ | 27.20 | 27.60 (+0.4000) | 26.60 (-0.6000) | 23.00 (-4.2000) |
| coco_cap_chair | coco_cap_recall ↑ | 60.11 | 61.88 (+1.7643) | 61.31 (+1.1972) | 63.01 (+2.8986) |
| coco_cap_chair | words per caption | 106.94 | 103.44 (-3.5040) | 107.33 (+0.3820) | 93.92 (-13.0280) |
| docci_caption | docci_Bleu_4 ↑ | 0.0477 | 0.0392 (-0.0085) | 0.0466 (-0.0011) | 0.0350 (-0.0128) |
| docci_caption | docci_CAPTURE ↑ | 0.5955 | 0.5885 (-0.0070) | 0.5930 (-0.0025) | 0.5855 (-0.0100) |
| docci_caption | docci_CIDEr ↑ | 0.1103 | 0.0952 (-0.0151) | 0.1168 (+0.0065) | 0.0988 (-0.0115) |
| docci_caption | docci_METEOR ↑ | 0.1378 | 0.1238 (-0.0139) | 0.1362 (-0.0015) | 0.1190 (-0.0188) |
| docci_caption | docci_ROUGE_L ↑ | 0.2261 | 0.2192 (-0.0069) | 0.2251 (-0.0010) | 0.2152 (-0.0109) |
| docci_caption | words per caption | 93.02 | 75.23 (-17.7900) | 93.16 (+0.1400) | 73.65 (-19.3640) |

## general (mmvp, mmbench_en_dev_static, mme, mmvet), gen_kwargs `temperature=0`

| task | metric | base | self | external | length |
|---|---|---|---|---|---|
| mmbench_en_dev_static | gpt_eval_score ↑ | 83.33 | 83.51 (+0.1718) | 83.16 (-0.1718) | 83.25 (-0.0859) |
| mme | mme_cognition_score ↑ | 618.57 | 621.07 (+2.5000) | 623.21 (+4.6429) | 625.36 (+6.7857) |
| mme | mme_perception_score ↑ | 1680.94 | 1690.00 (+9.0640) | 1686.11 (+5.1675) | 1676.26 (-4.6814) |
| mmvet | gpt_eval_score ↑ | 72.16 | 71.79 (-0.3670) | 69.98 (-2.1734) | 69.57 (-2.5862) |
| mmvp | mmvp_accuracy ↑ | 0.7833 | 0.7733 (-0.0100) | 0.7800 (-0.0033) | 0.7800 (-0.0033) |
| mmvp | mmvp_pair_accuracy ↑ | 0.5733 | 0.5600 (-0.0133) | 0.5667 (-0.0067) | 0.5667 (-0.0067) |

Result files:

- base/mcq: `lmms_eval_out/base/mcq/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261104_013053_results.json`
- base/caption: `lmms_eval_out/base/caption/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261106_182219_results.json`
- base/general: `lmms_eval_out/base/general/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261107_015704_results.json`
- self/mcq: `lmms_eval_out/self/mcq/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261104_020911_results.json`
- self/caption: `lmms_eval_out/self/caption/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261106_182229_results.json`
- self/general: `lmms_eval_out/self/general/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261107_020013_results.json`
- external/mcq: `lmms_eval_out/external/mcq/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261104_021127_results.json`
- external/caption: `lmms_eval_out/external/caption/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261106_220639_results.json`
- external/general: `lmms_eval_out/external/general/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261107_020417_results.json`
- length/mcq: `lmms_eval_out/length/mcq/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261104_021135_results.json`
- length/caption: `lmms_eval_out/length/caption/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261106_220910_results.json`
- length/general: `lmms_eval_out/length/general/snapshots__cc594898137f460bfe9f0759e9844b3ce807cfb5/20261107_020810_results.json`
