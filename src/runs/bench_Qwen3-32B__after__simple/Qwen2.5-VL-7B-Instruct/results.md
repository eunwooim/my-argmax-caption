# Qwen2.5-VL-7B-Instruct

46020 questions on 4985 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 90.0 [89.5, 90.5] (n=18172) |
| negative accuracy | 55.8 [55.0, 56.6] (n=18530) |
| negative: contradiction | 42.5 [41.0, 44.1] (n=4023) |
| negative: omission | 59.5 [58.6, 60.4] (n=14507) |
| hallucinated: truth accuracy | 26.2 [25.3, 27.2] (n=9318) |
| hallucinated: repeats Caption 2's claim ↓ | 38.5 [37.4, 39.7] (n=9318) |
| contradiction: repeats Caption 2's claim ↓ | 38.3 [36.4, 40.1] (n=2791) |
| positive + negative accuracy | 72.7 [72.3, 73.2] (n=36702) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (4096) | 89.8 | 54.7 | 41.8 | 58.9 | 26.2 | 38.6 |
| P+N (830) | 90.7 | 61.9 | 59.4 | 62.0 | – | – |
| P+H (28) | 92.3 | – | – | – | 33.3 | 33.3 |
| P (31) | 95.0 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 81.5 | 42.0 | 41.2 | 42.7 | 32.7 | 32.7 |
| attribute | 90.7 | 58.5 | 43.9 | 63.0 | 25.3 | 39.9 |
| count | 91.2 | 40.3 | 35.8 | 41.3 | 24.3 | 53.2 |
| object | 89.5 | 49.7 | 35.7 | 52.4 | 26.3 | 38.3 |
| relational | 100.0 | 50.0 | – | 50.0 | 20.0 | 60.0 |
| scene | 93.4 | 65.8 | 40.7 | 70.0 | 28.8 | 34.6 |
| spatial | 81.0 | 53.8 | 50.6 | 54.4 | 35.5 | 25.6 |
| text | 94.9 | 49.2 | 36.0 | 56.5 | 18.1 | 43.5 |
