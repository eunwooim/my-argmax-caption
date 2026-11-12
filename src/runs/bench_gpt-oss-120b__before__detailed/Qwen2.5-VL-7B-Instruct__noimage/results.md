# Qwen2.5-VL-7B-Instruct__noimage

35615 questions on 4982 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 48.1 [47.4, 48.8] (n=18122) |
| negative accuracy | 36.4 [35.6, 37.3] (n=12561) |
| negative: contradiction | 31.4 [29.9, 33.0] (n=3427) |
| negative: omission | 38.3 [37.3, 39.3] (n=9134) |
| hallucinated: truth accuracy | 28.0 [26.8, 29.2] (n=4932) |
| hallucinated: repeats Caption 2's claim ↓ | 13.6 [12.7, 14.6] (n=4932) |
| contradiction: repeats Caption 2's claim ↓ | 14.3 [12.3, 16.4] (n=1111) |
| positive + negative accuracy | 43.3 [42.8, 43.8] (n=30683) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (2888) | 48.0 | 35.5 | 30.7 | 37.6 | 28.0 | 13.7 |
| P+N (1868) | 48.1 | 38.0 | 33.2 | 39.3 | – | – |
| P+H (127) | 47.4 | – | – | – | 26.6 | 12.5 |
| P (99) | 48.0 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 51.9 | 43.3 | 41.6 | 45.2 | 37.7 | 17.2 |
| attribute | 48.0 | 35.7 | 30.6 | 38.5 | 27.3 | 15.2 |
| count | 44.5 | 29.2 | 28.1 | 29.5 | 24.3 | 28.4 |
| object | 47.5 | 36.6 | 29.1 | 38.0 | 21.0 | 13.4 |
| relational | 47.0 | 47.1 | 80.0 | 33.3 | 46.2 | 15.4 |
| scene | 57.2 | 53.4 | 55.4 | 52.2 | 30.3 | 11.8 |
| spatial | 53.3 | 46.4 | 37.4 | 48.8 | 41.2 | 12.2 |
| text | 28.6 | 24.0 | 27.5 | 21.9 | 46.3 | 4.0 |
