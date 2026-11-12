# Qwen2.5-VL-7B-Instruct

57673 questions on 4997 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 89.2 [88.7, 89.6] (n=23248) |
| negative accuracy | 52.8 [52.0, 53.6] (n=19770) |
| negative: contradiction | 42.6 [41.2, 43.9] (n=5741) |
| negative: omission | 57.1 [56.1, 58.0] (n=14029) |
| hallucinated: truth accuracy | 26.7 [25.9, 27.5] (n=14655) |
| hallucinated: repeats Caption 2's claim ↓ | 35.6 [34.7, 36.4] (n=14655) |
| contradiction: repeats Caption 2's claim ↓ | 34.3 [32.9, 35.8] (n=4225) |
| positive + negative accuracy | 72.5 [72.0, 72.9] (n=43018) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (4592) | 89.1 | 52.2 | 42.4 | 56.5 | 26.7 | 35.6 |
| P+N (366) | 90.4 | 62.0 | 51.0 | 62.9 | – | – |
| P+H (26) | 87.7 | – | – | – | 22.4 | 26.9 |
| P (13) | 86.2 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 83.8 | 43.2 | 38.0 | 47.9 | 30.4 | 29.8 |
| attribute | 89.3 | 54.4 | 44.0 | 59.1 | 27.2 | 35.4 |
| count | 88.5 | 37.7 | 32.6 | 39.7 | 24.2 | 41.9 |
| object | 90.8 | 48.5 | 34.8 | 52.6 | 24.8 | 40.0 |
| relational | 90.0 | 60.0 | – | 60.0 | 35.0 | 40.0 |
| scene | 92.3 | 62.4 | 45.9 | 67.9 | 25.8 | 33.9 |
| spatial | 77.8 | 54.6 | 51.5 | 55.3 | 35.2 | 22.0 |
| text | 93.2 | 43.7 | 31.5 | 49.9 | 15.9 | 42.9 |
