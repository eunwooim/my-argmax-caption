# DOCCI Positive / Negative / Hallucinated QA

- generator: openai/gpt-oss-120b; Caption 2: Qwen/Qwen2.5-VL-7B-Instruct (before RL, LoRA off) (simple prompt); format: open-ended
- images: 5000 in, **4971 kept** (at least one positive QA); dropped: {'no_positive': 29}
- QA: **36458** = positive 17142 / negative 13612 (contradiction 2915, omission 10697) / hallucinated 5704
- contradictions linked to their hallucinated QA: 1279; questions whose meta-language was removed: 5225
- judges: gpt-oss-120b (require all)

## Coverage (which sets each kept image has)

| case | meaning | images | % | positive QA | negative QA | hallucinated QA |
|---|---|---|---|---|---|---|
| P+N+H | all three sets | 3447 | 69.3 | 11821 | 9886 | 5556 |
| P+N | positive + negative | 1365 | 27.5 | 4778 | 3726 | 0 |
| P+H | positive + hallucinated | 102 | 2.1 | 346 | 0 | 148 |
| P | positive only | 57 | 1.1 | 197 | 0 | 0 |

Dropped for having no positive QA, by what they had: {'N': 12, 'N+H': 16, 'none': 1}

## Dropped QA

| reason | n |
|---|---|
| hallucinated: rule truth_too_long | 3230 |
| hallucinated: rule meta_language | 2488 |
| positive: judge (wrong set / answer) | 2418 |
| negative: rule answer_too_long | 1893 |
| negative: judge (wrong set / answer) | 1286 |
| negative: rule meta_language | 1186 |
| hallucinated: rule answer_too_long | 1151 |
| hallucinated: judge (wrong set / answer) | 983 |
| positive: rule answer_too_long | 384 |
| positive: rule meta_language | 185 |
| negative: rule bad_category | 104 |
| positive: rule bad_category | 96 |
| hallucinated: rule bad_category | 89 |
| positive: judge (not atomic) | 10 |
| hallucinated: judge (not atomic) | 7 |
| positive: rule not_a_question | 6 |
| positive: rule duplicate | 6 |
| hallucinated: rule claim_equals_truth | 2 |
| hallucinated: rule not_a_question | 1 |
| negative: judge (not atomic) | 1 |
| hallucinated: rule duplicate | 1 |
| negative: rule duplicate | 1 |

## Category

| category | positive | negative | hallucinated |
|---|---|---|---|
| object | 7756 | 4470 | 1964 |
| attribute | 5026 | 6251 | 2554 |
| action | 369 | 264 | 183 |
| count | 599 | 985 | 141 |
| spatial | 681 | 1072 | 176 |
| scene | 1889 | 178 | 204 |
| text | 778 | 377 | 463 |
| relational | 44 | 15 | 19 |
