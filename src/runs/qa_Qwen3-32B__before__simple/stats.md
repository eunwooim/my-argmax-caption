# DOCCI Positive / Negative / Hallucinated QA

- generator: Qwen/Qwen3-32B; Caption 2: Qwen/Qwen2.5-VL-7B-Instruct (before RL, LoRA off) (simple prompt); format: open-ended
- images: 5000 in, **4991 kept** (at least one positive QA); dropped: {'no_positive': 9}
- QA: **48333** = positive 18997 / negative 18272 (contradiction 4399, omission 13873) / hallucinated 11064
- contradictions linked to their hallucinated QA: 3020; questions whose meta-language was removed: 11652
- judges: Qwen3-32B (require all)

## Coverage (which sets each kept image has)

| case | meaning | images | % | positive QA | negative QA | hallucinated QA |
|---|---|---|---|---|---|---|
| P+N+H | all three sets | 4314 | 86.4 | 16572 | 16236 | 10982 |
| P+N | positive + negative | 615 | 12.3 | 2229 | 2036 | 0 |
| P+H | positive + hallucinated | 38 | 0.8 | 121 | 0 | 82 |
| P | positive only | 24 | 0.5 | 75 | 0 | 0 |

Dropped for having no positive QA, by what they had: {'N+H': 5, 'N': 4}

## Dropped QA

| reason | n |
|---|---|
| hallucinated: rule answer_too_long | 3774 |
| negative: rule answer_too_long | 3300 |
| hallucinated: rule truth_too_long | 2032 |
| negative: judge (wrong set / answer) | 1217 |
| hallucinated: rule meta_language | 1193 |
| positive: rule answer_too_long | 883 |
| positive: judge (wrong set / answer) | 275 |
| negative: rule duplicate | 193 |
| hallucinated: rule claim_equals_truth | 187 |
| negative: rule meta_language | 187 |
| positive: rule meta_language | 152 |
| hallucinated: rule duplicate | 117 |
| hallucinated: judge (wrong set / answer) | 93 |
| positive: rule duplicate | 6 |
| positive: judge (not atomic) | 3 |
| negative: judge (not atomic) | 3 |
| hallucinated: judge (not atomic) | 1 |

## Category

| category | positive | negative | hallucinated |
|---|---|---|---|
| object | 4287 | 2912 | 1830 |
| attribute | 8003 | 11643 | 6541 |
| action | 591 | 332 | 402 |
| count | 457 | 630 | 93 |
| spatial | 1142 | 1643 | 376 |
| scene | 3765 | 646 | 1313 |
| text | 746 | 464 | 494 |
| relational | 6 | 2 | 15 |
