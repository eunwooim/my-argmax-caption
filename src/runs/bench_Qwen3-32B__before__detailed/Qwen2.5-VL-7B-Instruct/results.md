# Qwen2.5-VL-7B-Instruct

59325 questions on 5000 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 88.3 [87.9, 88.8] (n=23579) |
| negative accuracy | 51.1 [50.3, 51.9] (n=19674) |
| negative: contradiction | 43.3 [42.0, 44.5] (n=6239) |
| negative: omission | 54.8 [53.8, 55.8] (n=13435) |
| hallucinated: truth accuracy | 27.1 [26.4, 27.9] (n=16072) |
| hallucinated: repeats Caption 2's claim ↓ | 33.9 [33.0, 34.7] (n=16072) |
| contradiction: repeats Caption 2's claim ↓ | 34.2 [32.8, 35.6] (n=4673) |
| positive + negative accuracy | 71.4 [70.9, 71.9] (n=43253) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (4661) | 88.1 | 50.7 | 43.1 | 54.5 | 27.2 | 33.9 |
| P+N (274) | 91.9 | 59.4 | 53.2 | 60.0 | – | – |
| P+H (51) | 89.5 | – | – | – | 24.3 | 33.8 |
| P (14) | 90.6 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 81.8 | 45.7 | 42.7 | 47.8 | 35.4 | 28.4 |
| attribute | 88.2 | 52.7 | 44.9 | 56.8 | 27.4 | 33.8 |
| count | 86.4 | 37.1 | 33.9 | 38.4 | 27.0 | 44.0 |
| object | 89.9 | 47.2 | 37.6 | 50.6 | 26.6 | 37.8 |
| relational | 100.0 | 55.6 | – | 55.6 | 25.0 | 30.6 |
| scene | 91.5 | 58.5 | 45.5 | 64.1 | 25.7 | 30.3 |
| spatial | 77.0 | 50.9 | 43.0 | 52.8 | 35.5 | 22.1 |
| text | 93.3 | 43.6 | 34.3 | 49.7 | 14.9 | 42.2 |
