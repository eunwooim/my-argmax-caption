# Qwen2.5-VL-7B-Instruct

36483 questions on 4988 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 95.6 [95.3, 95.9] (n=18357) |
| negative accuracy | 52.8 [51.9, 53.7] (n=12885) |
| negative: contradiction | 42.6 [40.8, 44.3] (n=3287) |
| negative: omission | 56.3 [55.3, 57.3] (n=9598) |
| hallucinated: truth accuracy | 21.6 [20.4, 22.7] (n=5241) |
| hallucinated: repeats Caption 2's claim ↓ | 45.4 [44.1, 46.8] (n=5241) |
| contradiction: repeats Caption 2's claim ↓ | 43.7 [40.9, 46.5] (n=1215) |
| positive + negative accuracy | 78.0 [77.5, 78.4] (n=31242) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (3108) | 95.6 | 51.7 | 40.7 | 56.3 | 21.8 | 45.4 |
| P+N (1667) | 95.7 | 54.9 | 48.7 | 56.3 | – | – |
| P+H (123) | 96.0 | – | – | – | 15.3 | 46.0 |
| P (90) | 95.9 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 92.4 | 43.6 | 38.4 | 49.6 | 29.7 | 36.1 |
| attribute | 95.1 | 53.5 | 43.8 | 58.4 | 25.0 | 46.2 |
| count | 95.3 | 41.0 | 36.0 | 42.5 | 27.7 | 52.4 |
| object | 96.2 | 54.2 | 41.1 | 56.4 | 17.7 | 44.5 |
| relational | 95.6 | 45.0 | 38.5 | 57.1 | 20.0 | 40.0 |
| scene | 97.1 | 59.3 | 36.8 | 72.8 | 27.1 | 45.0 |
| spatial | 91.0 | 58.3 | 49.2 | 60.5 | 34.2 | 33.5 |
| text | 95.8 | 41.6 | 40.3 | 42.4 | 8.9 | 49.6 |
