# DOCCI Positive / Negative / Hallucinated QA

- generator: openai/gpt-oss-120b; Caption 2: Qwen/Qwen2.5-VL-7B-Instruct + GRPO LoRA (/mnt/data1/eunwooim/argmax_caption/out_grpo_tok4096/adapter) (simple prompt); format: open-ended
- images: 5000 in, **4973 kept** (at least one positive QA); dropped: {'no_positive': 27}
- QA: **36632** = positive 16811 / negative 14352 (contradiction 2790, omission 11562) / hallucinated 5469
- contradictions linked to their hallucinated QA: 1298; questions whose meta-language was removed: 5113
- judges: gpt-oss-120b (require all)

## Coverage (which sets each kept image has)

| case | meaning | images | % | positive QA | negative QA | hallucinated QA |
|---|---|---|---|---|---|---|
| P+N+H | all three sets | 3371 | 67.8 | 11344 | 10119 | 5367 |
| P+N | positive + negative | 1483 | 29.8 | 5070 | 4233 | 0 |
| P+H | positive + hallucinated | 76 | 1.5 | 246 | 0 | 102 |
| P | positive only | 43 | 0.9 | 151 | 0 | 0 |

Dropped for having no positive QA, by what they had: {'N+H': 18, 'N': 9}

## Dropped QA

| reason | n |
|---|---|
| hallucinated: rule truth_too_long | 3060 |
| positive: judge (wrong set / answer) | 2362 |
| hallucinated: rule meta_language | 2108 |
| negative: rule answer_too_long | 1848 |
| negative: judge (wrong set / answer) | 1167 |
| negative: rule meta_language | 1059 |
| hallucinated: rule answer_too_long | 996 |
| hallucinated: judge (wrong set / answer) | 970 |
| positive: rule answer_too_long | 372 |
| positive: rule meta_language | 182 |
| positive: rule bad_category | 114 |
| hallucinated: rule bad_category | 103 |
| negative: rule bad_category | 101 |
| positive: judge (not atomic) | 16 |
| positive: rule not_a_question | 13 |
| hallucinated: judge (not atomic) | 6 |
| negative: judge (not atomic) | 4 |
| hallucinated: rule claim_equals_truth | 3 |
| negative: rule duplicate | 3 |
| hallucinated: rule not_a_question | 1 |
| negative: rule not_a_question | 1 |

## Category

| category | positive | negative | hallucinated |
|---|---|---|---|
| object | 7649 | 4871 | 1854 |
| attribute | 4970 | 6464 | 2548 |
| action | 402 | 247 | 205 |
| count | 604 | 985 | 139 |
| spatial | 753 | 1146 | 194 |
| scene | 1633 | 228 | 130 |
| text | 745 | 392 | 373 |
| relational | 55 | 19 | 26 |
