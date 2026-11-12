# Qwen2.5-VL-7B-Instruct__noimage

36458 questions on 4971 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 43.9 [43.2, 44.6] (n=17142) |
| negative accuracy | 37.4 [36.6, 38.2] (n=13612) |
| negative: contradiction | 29.8 [28.2, 31.5] (n=2915) |
| negative: omission | 39.5 [38.5, 40.4] (n=10697) |
| hallucinated: truth accuracy | 26.7 [25.6, 27.8] (n=5704) |
| hallucinated: repeats Caption 2's claim ↓ | 13.6 [12.7, 14.5] (n=5704) |
| contradiction: repeats Caption 2's claim ↓ | 13.9 [12.0, 15.8] (n=1279) |
| positive + negative accuracy | 41.0 [40.5, 41.6] (n=30754) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (3447) | 44.0 | 36.8 | 29.1 | 39.5 | 26.6 | 13.7 |
| P+N (1365) | 43.7 | 38.9 | 33.8 | 39.5 | – | – |
| P+H (102) | 43.4 | – | – | – | 29.1 | 9.5 |
| P (57) | 42.6 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 54.7 | 41.3 | 40.7 | 41.9 | 32.2 | 19.1 |
| attribute | 47.3 | 36.4 | 28.3 | 39.5 | 25.3 | 15.6 |
| count | 42.2 | 32.5 | 27.3 | 33.4 | 21.3 | 29.1 |
| object | 39.6 | 37.2 | 26.3 | 38.9 | 24.3 | 11.9 |
| relational | 56.8 | 53.3 | 66.7 | 44.4 | 47.4 | 5.3 |
| scene | 57.6 | 50.0 | 46.0 | 51.6 | 31.4 | 14.2 |
| spatial | 51.8 | 49.1 | 49.7 | 49.0 | 33.0 | 18.2 |
| text | 19.8 | 26.3 | 22.0 | 28.0 | 38.9 | 0.9 |
