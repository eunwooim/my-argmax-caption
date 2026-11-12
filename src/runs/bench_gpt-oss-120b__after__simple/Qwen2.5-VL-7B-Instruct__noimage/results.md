# Qwen2.5-VL-7B-Instruct__noimage

36632 questions on 4973 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 43.5 [42.8, 44.3] (n=16811) |
| negative accuracy | 37.8 [36.9, 38.5] (n=14352) |
| negative: contradiction | 31.1 [29.4, 32.8] (n=2790) |
| negative: omission | 39.4 [38.4, 40.3] (n=11562) |
| hallucinated: truth accuracy | 27.2 [26.0, 28.4] (n=5469) |
| hallucinated: repeats Caption 2's claim ↓ | 14.2 [13.3, 15.2] (n=5469) |
| contradiction: repeats Caption 2's claim ↓ | 14.5 [12.6, 16.4] (n=1298) |
| positive + negative accuracy | 40.9 [40.3, 41.4] (n=31163) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (3371) | 43.1 | 37.3 | 30.9 | 39.3 | 27.2 | 14.3 |
| P+N (1483) | 44.2 | 38.8 | 32.4 | 39.4 | – | – |
| P+H (76) | 46.3 | – | – | – | 27.5 | 11.8 |
| P (43) | 46.4 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 53.0 | 39.7 | 39.1 | 40.4 | 36.6 | 20.5 |
| attribute | 46.9 | 36.4 | 29.9 | 38.5 | 26.3 | 15.9 |
| count | 40.7 | 31.8 | 34.8 | 31.3 | 22.3 | 20.1 |
| object | 38.8 | 38.6 | 27.6 | 40.1 | 23.6 | 12.8 |
| relational | 58.2 | 52.6 | 66.7 | 40.0 | 30.8 | 23.1 |
| scene | 57.8 | 45.6 | 35.6 | 49.1 | 31.5 | 10.8 |
| spatial | 57.4 | 48.3 | 44.9 | 49.0 | 37.1 | 17.5 |
| text | 20.7 | 26.3 | 25.2 | 26.7 | 40.8 | 2.9 |
