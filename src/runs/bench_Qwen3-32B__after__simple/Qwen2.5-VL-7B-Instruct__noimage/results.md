# Qwen2.5-VL-7B-Instruct__noimage

46020 questions on 4985 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 44.0 [43.3, 44.7] (n=18172) |
| negative accuracy | 38.7 [38.0, 39.4] (n=18530) |
| negative: contradiction | 31.2 [29.8, 32.7] (n=4023) |
| negative: omission | 40.8 [40.0, 41.6] (n=14507) |
| hallucinated: truth accuracy | 32.3 [31.3, 33.3] (n=9318) |
| hallucinated: repeats Caption 2's claim ↓ | 12.4 [11.7, 13.1] (n=9318) |
| contradiction: repeats Caption 2's claim ↓ | 12.0 [10.8, 13.3] (n=2791) |
| positive + negative accuracy | 41.3 [40.8, 41.8] (n=36702) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (4096) | 43.7 | 38.4 | 31.4 | 40.7 | 32.4 | 12.3 |
| P+N (830) | 45.4 | 40.4 | 26.9 | 41.2 | – | – |
| P+H (28) | 40.7 | – | – | – | 9.5 | 26.2 |
| P (31) | 41.0 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 46.6 | 39.1 | 35.3 | 42.2 | 36.1 | 11.6 |
| attribute | 49.9 | 38.6 | 29.6 | 41.4 | 28.4 | 15.0 |
| count | 29.0 | 26.6 | 21.1 | 27.7 | 19.8 | 26.1 |
| object | 35.5 | 33.4 | 26.1 | 34.8 | 32.9 | 8.1 |
| relational | 50.0 | 50.0 | – | 50.0 | 60.0 | 0.0 |
| scene | 44.9 | 59.5 | 49.1 | 61.3 | 46.8 | 6.1 |
| spatial | 51.0 | 46.5 | 46.9 | 46.4 | 39.8 | 12.8 |
| text | 22.5 | 32.6 | 41.5 | 27.8 | 51.1 | 1.2 |
