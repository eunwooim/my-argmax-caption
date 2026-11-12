# Qwen2.5-VL-7B-Instruct__noimage

59325 questions on 5000 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 44.8 [44.1, 45.4] (n=23579) |
| negative accuracy | 38.5 [37.8, 39.2] (n=19674) |
| negative: contradiction | 33.7 [32.5, 34.9] (n=6239) |
| negative: omission | 40.7 [39.8, 41.5] (n=13435) |
| hallucinated: truth accuracy | 34.1 [33.3, 35.0] (n=16072) |
| hallucinated: repeats Caption 2's claim ↓ | 12.0 [11.4, 12.5] (n=16072) |
| contradiction: repeats Caption 2's claim ↓ | 9.9 [9.0, 10.8] (n=4673) |
| positive + negative accuracy | 41.9 [41.4, 42.4] (n=43253) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (4661) | 44.7 | 38.4 | 33.7 | 40.7 | 34.1 | 12.0 |
| P+N (274) | 44.8 | 39.6 | 34.2 | 40.1 | – | – |
| P+H (51) | 48.4 | – | – | – | 44.1 | 11.0 |
| P (14) | 50.0 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 53.0 | 42.1 | 39.5 | 43.9 | 42.5 | 11.9 |
| attribute | 51.2 | 38.3 | 32.6 | 41.3 | 28.7 | 13.8 |
| count | 27.7 | 25.8 | 22.2 | 27.3 | 23.4 | 25.8 |
| object | 39.5 | 32.9 | 29.7 | 34.0 | 37.6 | 9.7 |
| relational | 90.0 | 55.6 | – | 55.6 | 55.6 | 8.3 |
| scene | 36.7 | 53.6 | 49.3 | 55.4 | 54.4 | 6.1 |
| spatial | 50.1 | 45.7 | 44.3 | 46.0 | 39.5 | 8.7 |
| text | 28.8 | 38.5 | 44.6 | 34.6 | 50.5 | 2.3 |
