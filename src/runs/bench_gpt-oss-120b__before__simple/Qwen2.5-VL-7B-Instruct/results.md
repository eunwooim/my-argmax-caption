# Qwen2.5-VL-7B-Instruct

36458 questions on 4971 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 95.5 [95.2, 95.8] (n=17142) |
| negative accuracy | 55.1 [54.3, 56.0] (n=13612) |
| negative: contradiction | 42.0 [40.1, 43.8] (n=2915) |
| negative: omission | 58.7 [57.7, 59.6] (n=10697) |
| hallucinated: truth accuracy | 22.9 [21.8, 24.0] (n=5704) |
| hallucinated: repeats Caption 2's claim ↓ | 43.0 [41.7, 44.3] (n=5704) |
| contradiction: repeats Caption 2's claim ↓ | 45.3 [42.6, 48.1] (n=1279) |
| positive + negative accuracy | 77.6 [77.2, 78.1] (n=30754) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (3447) | 95.3 | 53.9 | 40.9 | 58.4 | 22.8 | 43.0 |
| P+N (1365) | 95.9 | 58.3 | 49.4 | 59.4 | – | – |
| P+H (102) | 94.8 | – | – | – | 27.0 | 41.2 |
| P (57) | 96.4 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 91.6 | 43.9 | 39.3 | 48.8 | 30.1 | 35.0 |
| attribute | 95.6 | 56.6 | 42.0 | 62.0 | 26.6 | 42.6 |
| count | 94.2 | 43.4 | 40.6 | 43.8 | 28.4 | 53.9 |
| object | 95.7 | 54.9 | 41.7 | 57.1 | 20.1 | 41.8 |
| relational | 95.5 | 53.3 | 33.3 | 66.7 | 26.3 | 42.1 |
| scene | 96.4 | 66.3 | 40.0 | 76.6 | 26.5 | 42.6 |
| spatial | 91.3 | 61.8 | 50.9 | 63.8 | 26.1 | 42.6 |
| text | 96.3 | 47.2 | 37.6 | 51.1 | 7.6 | 50.3 |
