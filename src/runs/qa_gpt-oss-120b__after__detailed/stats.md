# DOCCI Positive / Negative / Hallucinated QA

- generator: openai/gpt-oss-120b; Caption 2: Qwen/Qwen2.5-VL-7B-Instruct + GRPO LoRA (/mnt/data1/eunwooim/argmax_caption/out_grpo_tok4096/adapter) (detailed prompt); format: open-ended
- images: 5000 in, **4988 kept** (at least one positive QA); dropped: {'no_positive': 12}
- QA: **36483** = positive 18357 / negative 12885 (contradiction 3287, omission 9598) / hallucinated 5241
- contradictions linked to their hallucinated QA: 1215; questions whose meta-language was removed: 4600
- judges: gpt-oss-120b (require all)

## Coverage (which sets each kept image has)

| case | meaning | images | % | positive QA | negative QA | hallucinated QA |
|---|---|---|---|---|---|---|
| P+N+H | all three sets | 3108 | 62.3 | 11487 | 8626 | 5078 |
| P+N | positive + negative | 1667 | 33.4 | 6102 | 4259 | 0 |
| P+H | positive + hallucinated | 123 | 2.5 | 425 | 0 | 163 |
| P | positive only | 90 | 1.8 | 343 | 0 | 0 |

Dropped for having no positive QA, by what they had: {'N': 8, 'N+H': 4}

## Dropped QA

| reason | n |
|---|---|
| hallucinated: rule meta_language | 3391 |
| hallucinated: rule truth_too_long | 3246 |
| positive: judge (wrong set / answer) | 2609 |
| negative: rule answer_too_long | 1974 |
| negative: judge (wrong set / answer) | 1756 |
| negative: rule meta_language | 1270 |
| hallucinated: rule answer_too_long | 1163 |
| hallucinated: judge (wrong set / answer) | 1017 |
| positive: rule answer_too_long | 412 |
| hallucinated: rule bad_category | 244 |
| negative: rule bad_category | 192 |
| positive: rule bad_category | 192 |
| positive: rule meta_language | 135 |
| positive: judge (not atomic) | 12 |
| positive: rule not_a_question | 10 |
| hallucinated: rule claim_equals_truth | 7 |
| negative: rule duplicate | 2 |
| hallucinated: judge (not atomic) | 2 |
| negative: judge (not atomic) | 1 |
| hallucinated: rule duplicate | 1 |

## Category

| category | positive | negative | hallucinated |
|---|---|---|---|
| object | 8492 | 4451 | 1738 |
| attribute | 4953 | 5689 | 2374 |
| action | 393 | 259 | 158 |
| count | 743 | 934 | 166 |
| spatial | 835 | 1021 | 155 |
| scene | 2045 | 182 | 129 |
| text | 828 | 329 | 506 |
| relational | 68 | 20 | 15 |
