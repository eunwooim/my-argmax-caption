# Qwen2.5-VL-7B-Instruct__noimage

36483 questions on 4988 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 47.1 [46.4, 47.9] (n=18357) |
| negative accuracy | 37.2 [36.3, 38.0] (n=12885) |
| negative: contradiction | 31.8 [30.2, 33.4] (n=3287) |
| negative: omission | 39.1 [38.1, 40.0] (n=9598) |
| hallucinated: truth accuracy | 26.4 [25.2, 27.6] (n=5241) |
| hallucinated: repeats Caption 2's claim ↓ | 15.3 [14.3, 16.3] (n=5241) |
| contradiction: repeats Caption 2's claim ↓ | 15.5 [13.5, 17.6] (n=1215) |
| positive + negative accuracy | 43.0 [42.5, 43.6] (n=31242) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (3108) | 46.6 | 36.6 | 31.0 | 38.9 | 26.4 | 15.4 |
| P+N (1667) | 47.8 | 38.4 | 34.2 | 39.3 | – | – |
| P+H (123) | 50.8 | – | – | – | 26.4 | 12.3 |
| P (90) | 48.7 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 54.7 | 37.8 | 37.7 | 38.0 | 35.4 | 17.7 |
| attribute | 48.5 | 35.7 | 29.7 | 38.7 | 24.3 | 17.2 |
| count | 42.7 | 31.5 | 27.5 | 32.6 | 22.9 | 24.7 |
| object | 45.9 | 37.7 | 28.2 | 39.4 | 23.0 | 15.2 |
| relational | 52.9 | 65.0 | 53.8 | 85.7 | 26.7 | 33.3 |
| scene | 55.5 | 51.6 | 54.4 | 50.0 | 31.8 | 11.6 |
| spatial | 53.2 | 48.4 | 51.3 | 47.7 | 34.8 | 17.4 |
| text | 25.4 | 27.1 | 36.1 | 21.9 | 42.5 | 2.6 |
