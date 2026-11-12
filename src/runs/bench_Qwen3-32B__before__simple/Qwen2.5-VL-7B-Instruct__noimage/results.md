# Qwen2.5-VL-7B-Instruct__noimage

48333 questions on 4991 images. Percent, 95% interval over images.

| metric | value |
|---|---|
| positive accuracy | 44.7 [44.0, 45.5] (n=18997) |
| negative accuracy | 38.7 [38.0, 39.5] (n=18272) |
| negative: contradiction | 32.9 [31.5, 34.4] (n=4399) |
| negative: omission | 40.6 [39.7, 41.4] (n=13873) |
| hallucinated: truth accuracy | 32.6 [31.7, 33.5] (n=11064) |
| hallucinated: repeats Caption 2's claim ↓ | 11.2 [10.6, 11.9] (n=11064) |
| contradiction: repeats Caption 2's claim ↓ | 10.7 [9.6, 11.8] (n=3020) |
| positive + negative accuracy | 41.8 [41.3, 42.3] (n=37269) |
| invalid replies | 0.0 |

## Per coverage case

| case (images) | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| P+N+H (4314) | 44.4 | 38.6 | 32.9 | 40.5 | 32.6 | 11.3 |
| P+N (615) | 46.4 | 40.1 | 34.4 | 40.7 | – | – |
| P+H (38) | 45.5 | – | – | – | 28.0 | 7.3 |
| P (24) | 56.0 | – | – | – | – | – |

## Per category

| category | positive | negative | contradiction | omission | hal: truth | hal: repeats ↓ |
|---|---|---|---|---|---|---|
| action | 52.3 | 38.0 | 26.5 | 44.2 | 35.3 | 12.2 |
| attribute | 50.5 | 39.6 | 33.4 | 41.9 | 28.6 | 13.8 |
| count | 33.0 | 25.1 | 21.5 | 25.8 | 21.5 | 19.4 |
| object | 35.0 | 32.2 | 25.6 | 33.8 | 34.3 | 6.3 |
| relational | 83.3 | 100.0 | – | 100.0 | 33.3 | 6.7 |
| scene | 47.1 | 50.0 | 44.4 | 51.6 | 43.4 | 8.1 |
| spatial | 48.4 | 46.0 | 43.2 | 46.4 | 36.4 | 10.6 |
| text | 21.3 | 34.1 | 37.8 | 32.0 | 47.2 | 1.6 |
