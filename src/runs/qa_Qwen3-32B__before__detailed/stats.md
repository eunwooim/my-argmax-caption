# DOCCI Positive / Negative / Hallucinated QA

- generator: Qwen/Qwen3-32B; Caption 2: Qwen/Qwen2.5-VL-7B-Instruct (before RL, LoRA off) (detailed prompt); format: open-ended
- images: 5000 in, **5000 kept** (at least one positive QA); dropped: none
- QA: **59325** = positive 23579 / negative 19674 (contradiction 6239, omission 13435) / hallucinated 16072
- contradictions linked to their hallucinated QA: 4673; questions whose meta-language was removed: 17503
- judges: Qwen3-32B (require all)

## Coverage (which sets each kept image has)

| case | meaning | images | % | positive QA | negative QA | hallucinated QA |
|---|---|---|---|---|---|---|
| P+N+H | all three sets | 4661 | 93.2 | 22162 | 18755 | 15936 |
| P+N | positive + negative | 274 | 5.5 | 1134 | 919 | 0 |
| P+H | positive + hallucinated | 51 | 1.0 | 219 | 0 | 136 |
| P | positive only | 14 | 0.3 | 64 | 0 | 0 |

## Dropped QA

| reason | n |
|---|---|
| hallucinated: rule answer_too_long | 4138 |
| negative: rule answer_too_long | 3307 |
| negative: judge (wrong set / answer) | 2800 |
| hallucinated: rule truth_too_long | 2078 |
| positive: rule answer_too_long | 960 |
| hallucinated: rule meta_language | 734 |
| negative: rule duplicate | 611 |
| hallucinated: rule claim_equals_truth | 600 |
| positive: judge (wrong set / answer) | 293 |
| hallucinated: rule duplicate | 293 |
| hallucinated: judge (wrong set / answer) | 230 |
| negative: rule meta_language | 151 |
| positive: rule meta_language | 119 |
| positive: rule duplicate | 6 |
| negative: judge (not atomic) | 3 |
| hallucinated: judge (not atomic) | 3 |
| positive: judge (not atomic) | 2 |

## Category

| category | positive | negative | hallucinated |
|---|---|---|---|
| object | 4986 | 2611 | 2021 |
| attribute | 10690 | 13140 | 10370 |
| action | 692 | 385 | 522 |
| count | 575 | 631 | 248 |
| spatial | 1667 | 1659 | 598 |
| scene | 4034 | 702 | 1611 |
| text | 925 | 537 | 666 |
| relational | 10 | 9 | 36 |
