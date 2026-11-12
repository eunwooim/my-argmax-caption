# DOCCI Positive / Negative / Hallucinated QA

- generator: Qwen/Qwen3-32B; Caption 2: Qwen/Qwen2.5-VL-7B-Instruct + GRPO LoRA (/mnt/data1/eunwooim/argmax_caption/out_grpo_tok4096/adapter) (simple prompt); format: open-ended
- images: 5000 in, **4985 kept** (at least one positive QA); dropped: {'json_parse_failure': 2, 'no_positive': 13}
- QA: **46020** = positive 18172 / negative 18530 (contradiction 4023, omission 14507) / hallucinated 9318
- contradictions linked to their hallucinated QA: 2791; questions whose meta-language was removed: 9924
- judges: Qwen3-32B (require all)

## Coverage (which sets each kept image has)

| case | meaning | images | % | positive QA | negative QA | hallucinated QA |
|---|---|---|---|---|---|---|
| P+N+H | all three sets | 4096 | 82.2 | 15023 | 15642 | 9276 |
| P+N | positive + negative | 830 | 16.6 | 2958 | 2888 | 0 |
| P+H | positive + hallucinated | 28 | 0.6 | 91 | 0 | 42 |
| P | positive only | 31 | 0.6 | 100 | 0 | 0 |

Dropped for having no positive QA, by what they had: {'N+H': 8, 'N': 5}

## Dropped QA

| reason | n |
|---|---|
| negative: rule answer_too_long | 3433 |
| hallucinated: rule answer_too_long | 3020 |
| hallucinated: rule truth_too_long | 1979 |
| hallucinated: rule meta_language | 1071 |
| negative: judge (wrong set / answer) | 1063 |
| positive: rule answer_too_long | 948 |
| positive: judge (wrong set / answer) | 281 |
| negative: rule meta_language | 171 |
| negative: rule duplicate | 165 |
| positive: rule meta_language | 145 |
| hallucinated: rule claim_equals_truth | 121 |
| hallucinated: judge (wrong set / answer) | 75 |
| hallucinated: rule duplicate | 74 |
| positive: rule duplicate | 8 |
| negative: judge (not atomic) | 4 |
| positive: judge (not atomic) | 4 |
| hallucinated: judge (not atomic) | 3 |

## Category

| category | positive | negative | hallucinated |
|---|---|---|---|
| object | 4404 | 3083 | 1613 |
| attribute | 7824 | 11691 | 5634 |
| action | 644 | 338 | 440 |
| count | 466 | 625 | 111 |
| spatial | 1283 | 1577 | 422 |
| scene | 2834 | 749 | 684 |
| text | 711 | 463 | 409 |
| relational | 6 | 4 | 5 |
