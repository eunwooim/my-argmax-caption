# Qwen2.5-VL-7B-Instruct

36632 questions on 4973 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 94.9 [94.5, 95.2] (n=16811) |
| negative accuracy | 56.2 [55.3, 57.0] (n=14352) |
| negative: contradiction | 40.6 [38.8, 42.4] (n=2790) |
| negative: omission | 59.9 [59.0, 60.9] (n=11562) |
| hallucinated: truth accuracy | 22.4 [21.3, 23.6] (n=5469) |
| hallucinated: repeats Caption 2's claim ↓ | 44.8 [43.5, 46.2] (n=5469) |
| contradiction: repeats Caption 2's claim ↓ | 45.6 [42.8, 48.3] (n=1298) |
| positive + negative accuracy | 77.1 [76.6, 77.5] (n=31163) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (3371) | 94.5 | 54.5 | 39.2 | 59.2 | 22.4 | 44.8 |
| P+N (1483) | 95.7 | 60.3 | 49.7 | 61.4 | – | – |
| P+H (76) | 93.9 | – | – | – | 23.5 | 45.1 |
| P (43) | 95.4 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 91.3 | 46.6 | 43.5 | 50.5 | 32.2 | 43.4 |
| attribute | 95.3 | 57.5 | 40.3 | 63.0 | 23.2 | 45.3 |
| count | 92.5 | 42.7 | 39.9 | 43.2 | 30.9 | 48.2 |
| object | 95.0 | 56.9 | 40.5 | 59.2 | 20.8 | 42.8 |
| relational | 89.1 | 36.8 | 33.3 | 40.0 | 38.5 | 38.5 |
| scene | 96.1 | 66.7 | 42.4 | 75.1 | 30.0 | 47.7 |
| spatial | 92.7 | 59.6 | 47.2 | 61.9 | 27.8 | 43.8 |
| text | 94.9 | 49.7 | 33.3 | 56.2 | 9.9 | 51.5 |
