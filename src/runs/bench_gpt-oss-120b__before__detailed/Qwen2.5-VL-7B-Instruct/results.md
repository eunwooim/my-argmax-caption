# Qwen2.5-VL-7B-Instruct

35615 questions on 4982 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 95.2 [94.9, 95.5] (n=18122) |
| negative accuracy | 52.2 [51.3, 53.1] (n=12561) |
| negative: contradiction | 41.3 [39.6, 42.9] (n=3427) |
| negative: omission | 56.2 [55.2, 57.3] (n=9134) |
| hallucinated: truth accuracy | 21.8 [20.6, 23.0] (n=4932) |
| hallucinated: repeats Caption 2's claim ↓ | 44.2 [42.9, 45.7] (n=4932) |
| contradiction: repeats Caption 2's claim ↓ | 46.6 [43.6, 49.6] (n=1111) |
| positive + negative accuracy | 77.6 [77.1, 78.0] (n=30683) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (2888) | 95.2 | 50.6 | 38.0 | 56.1 | 21.8 | 44.1 |
| P+N (1868) | 95.4 | 54.8 | 49.0 | 56.5 | – | – |
| P+H (127) | 94.8 | – | – | – | 23.4 | 48.9 |
| P (99) | 93.3 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 92.6 | 49.6 | 42.3 | 58.3 | 23.0 | 43.4 |
| attribute | 94.7 | 52.7 | 41.3 | 58.9 | 26.0 | 44.5 |
| count | 93.2 | 41.4 | 36.5 | 42.8 | 27.0 | 45.9 |
| object | 95.6 | 52.3 | 41.3 | 54.4 | 18.9 | 42.9 |
| relational | 89.4 | 70.6 | 80.0 | 66.7 | 30.8 | 30.8 |
| scene | 97.0 | 58.2 | 39.3 | 70.0 | 21.8 | 45.4 |
| spatial | 92.3 | 59.5 | 45.0 | 63.5 | 38.2 | 36.6 |
| text | 95.7 | 47.8 | 41.2 | 51.9 | 7.5 | 48.7 |
