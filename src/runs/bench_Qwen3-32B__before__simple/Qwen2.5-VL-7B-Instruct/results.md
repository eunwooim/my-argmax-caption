# Qwen2.5-VL-7B-Instruct

48333 questions on 4991 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 89.7 [89.3, 90.2] (n=18997) |
| negative accuracy | 54.5 [53.7, 55.2] (n=18272) |
| negative: contradiction | 43.8 [42.3, 45.3] (n=4399) |
| negative: omission | 57.9 [57.0, 58.7] (n=13873) |
| hallucinated: truth accuracy | 25.8 [24.9, 26.7] (n=11064) |
| hallucinated: repeats Caption 2's claim ↓ | 36.2 [35.3, 37.2] (n=11064) |
| contradiction: repeats Caption 2's claim ↓ | 36.6 [34.8, 38.3] (n=3020) |
| positive + negative accuracy | 72.4 [72.0, 72.9] (n=37269) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (4314) | 89.7 | 53.8 | 43.2 | 57.6 | 25.8 | 36.2 |
| P+N (615) | 90.3 | 59.4 | 57.4 | 59.6 | – | – |
| P+H (38) | 87.6 | – | – | – | 26.8 | 42.7 |
| P (24) | 94.7 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 84.1 | 45.8 | 41.9 | 47.9 | 32.1 | 29.6 |
| attribute | 90.2 | 56.9 | 44.6 | 61.3 | 26.0 | 37.3 |
| count | 92.1 | 41.0 | 36.4 | 41.9 | 32.3 | 50.5 |
| object | 88.1 | 47.9 | 39.3 | 50.0 | 25.4 | 36.7 |
| relational | 100.0 | 100.0 | – | 100.0 | 13.3 | 13.3 |
| scene | 92.9 | 58.0 | 40.3 | 63.1 | 23.6 | 32.2 |
| spatial | 80.5 | 55.3 | 51.4 | 55.9 | 36.2 | 25.3 |
| text | 94.6 | 50.9 | 43.3 | 55.0 | 17.4 | 42.5 |
