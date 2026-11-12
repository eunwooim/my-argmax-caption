# Qwen2.5-VL-7B-Instruct__noimage

57673 questions on 4997 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 44.6 [44.0, 45.3] (n=23248) |
| negative accuracy | 39.3 [38.6, 40.1] (n=19770) |
| negative: contradiction | 32.6 [31.3, 33.8] (n=5741) |
| negative: omission | 42.1 [41.2, 43.0] (n=14029) |
| hallucinated: truth accuracy | 34.1 [33.2, 34.9] (n=14655) |
| hallucinated: repeats Caption 2's claim ↓ | 12.5 [12.0, 13.1] (n=14655) |
| contradiction: repeats Caption 2's claim ↓ | 11.2 [10.2, 12.2] (n=4225) |
| positive + negative accuracy | 42.2 [41.7, 42.7] (n=43018) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (4592) | 44.6 | 39.2 | 32.7 | 42.1 | 34.0 | 12.5 |
| P+N (366) | 45.4 | 40.9 | 26.5 | 42.2 | – | – |
| P+H (26) | 41.5 | – | – | – | 41.8 | 13.4 |
| P (13) | 44.8 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 51.7 | 43.5 | 33.2 | 52.6 | 37.7 | 11.3 |
| attribute | 51.1 | 38.9 | 31.4 | 42.3 | 28.9 | 14.4 |
| count | 33.2 | 25.3 | 17.7 | 28.3 | 23.8 | 19.0 |
| object | 41.0 | 34.8 | 29.7 | 36.4 | 39.0 | 10.5 |
| relational | 65.0 | 80.0 | – | 80.0 | 50.0 | 10.0 |
| scene | 32.9 | 58.3 | 51.9 | 60.4 | 56.2 | 5.3 |
| spatial | 51.3 | 46.7 | 46.0 | 46.8 | 40.1 | 11.3 |
| text | 29.5 | 37.6 | 41.4 | 35.7 | 51.7 | 3.5 |
