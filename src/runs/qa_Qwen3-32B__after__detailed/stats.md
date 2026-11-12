# DOCCI Positive / Negative / Hallucinated QA

- generator: Qwen/Qwen3-32B; Caption 2: Qwen/Qwen2.5-VL-7B-Instruct + GRPO LoRA (/mnt/data1/eunwooim/argmax_caption/out_grpo_tok4096/adapter) (detailed prompt); format: open-ended
- images: 5000 in, **4997 kept** (at least one positive QA); dropped: {'no_positive': 2, 'json_parse_failure': 1}
- QA: **57673** = positive 23248 / negative 19770 (contradiction 5741, omission 14029) / hallucinated 14655
- contradictions linked to their hallucinated QA: 4225; questions whose meta-language was removed: 16197
- judges: Qwen3-32B (require all)

## Coverage (which sets each kept image has)

| case | meaning | images | % | positive QA | negative QA | hallucinated QA |
|---|---|---|---|---|---|---|
| P+N+H | all three sets | 4592 | 91.9 | 21471 | 18550 | 14588 |
| P+N | positive + negative | 366 | 7.3 | 1613 | 1220 | 0 |
| P+H | positive + hallucinated | 26 | 0.5 | 106 | 0 | 67 |
| P | positive only | 13 | 0.3 | 58 | 0 | 0 |

Dropped for having no positive QA, by what they had: {'N+H': 2}

## Dropped QA

| reason | n |
|---|---|
| hallucinated: rule answer_too_long | 3875 |
| negative: rule answer_too_long | 3471 |
| negative: judge (wrong set / answer) | 2438 |
| hallucinated: rule truth_too_long | 2191 |
| positive: rule answer_too_long | 994 |
| hallucinated: rule meta_language | 668 |
| hallucinated: rule claim_equals_truth | 495 |
| negative: rule duplicate | 457 |
| hallucinated: rule duplicate | 280 |
| positive: judge (wrong set / answer) | 262 |
| hallucinated: judge (wrong set / answer) | 202 |
| negative: rule meta_language | 160 |
| positive: rule meta_language | 154 |
| negative: judge (not atomic) | 2 |
| positive: judge (not atomic) | 1 |
| hallucinated: judge (not atomic) | 1 |

## Category

| category | positive | negative | hallucinated |
|---|---|---|---|
| object | 5300 | 2702 | 2005 |
| attribute | 10221 | 13042 | 9494 |
| action | 764 | 400 | 504 |
| count | 671 | 647 | 248 |
| spatial | 1756 | 1691 | 619 |
| scene | 3661 | 743 | 1161 |
| text | 855 | 540 | 604 |
| relational | 20 | 5 | 20 |
