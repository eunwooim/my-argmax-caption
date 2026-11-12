# DOCCI Positive / Negative / Hallucinated QA

- generator: openai/gpt-oss-120b; Caption 2: Qwen/Qwen2.5-VL-7B-Instruct (before RL, LoRA off) (detailed prompt); format: open-ended
- images: 5000 in, **4982 kept** (at least one positive QA); dropped: {'no_positive': 18}
- QA: **35615** = positive 18122 / negative 12561 (contradiction 3427, omission 9134) / hallucinated 4932
- contradictions linked to their hallucinated QA: 1111; questions whose meta-language was removed: 4276
- judges: gpt-oss-120b (require all)

## Coverage (which sets each kept image has)

| case | meaning | images | % | positive QA | negative QA | hallucinated QA |
|---|---|---|---|---|---|---|
| P+N+H | all three sets | 2888 | 58.0 | 10605 | 7847 | 4748 |
| P+N | positive + negative | 1868 | 37.5 | 6699 | 4714 | 0 |
| P+H | positive + hallucinated | 127 | 2.5 | 460 | 0 | 184 |
| P | positive only | 99 | 2.0 | 358 | 0 | 0 |

Dropped for having no positive QA, by what they had: {'N': 8, 'N+H': 9, 'none': 1}

## Dropped QA

| reason | n |
|---|---|
| hallucinated: rule meta_language | 4067 |
| hallucinated: rule truth_too_long | 3165 |
| positive: judge (wrong set / answer) | 2588 |
| negative: rule answer_too_long | 1889 |
| negative: judge (wrong set / answer) | 1768 |
| negative: rule meta_language | 1366 |
| hallucinated: rule answer_too_long | 1240 |
| hallucinated: judge (wrong set / answer) | 961 |
| positive: rule answer_too_long | 410 |
| hallucinated: rule bad_category | 200 |
| negative: rule bad_category | 170 |
| positive: rule bad_category | 132 |
| positive: rule meta_language | 130 |
| hallucinated: judge (not atomic) | 10 |
| positive: judge (not atomic) | 9 |
| positive: rule not_a_question | 5 |
| hallucinated: rule claim_equals_truth | 3 |
| negative: judge (not atomic) | 2 |
| negative: rule duplicate | 2 |
| negative: rule not_a_question | 1 |

## Category

| category | positive | negative | hallucinated |
|---|---|---|---|
| object | 8476 | 4148 | 1632 |
| attribute | 4902 | 5772 | 2223 |
| action | 339 | 252 | 122 |
| count | 696 | 910 | 148 |
| spatial | 700 | 975 | 131 |
| scene | 2087 | 146 | 119 |
| text | 856 | 341 | 544 |
| relational | 66 | 17 | 13 |
