# Reward comparison: test-time scaling and GRPO

Same benchmarks for both uses: CaptionQA (score, and agreement of the caption-only answers with the captioning model's own image answers), Self-Validation's CHAIR-500 (CHAIR and object F1), AMBER, caption length. Captions have no length cap (4096 tokens). Values: point estimate [95% bootstrap interval over images]. ✓ / ✗: the paired interval of ours − other excludes 0 in our favour / against us.

> Agreement not available for: self, random, length, selfcert, longclip, external (tts.py vqa not run).

## Use 1: test-time scaling (base model, no training)

The base model samples 16 captions per image (Self-Validation's sampling: temperature 0.5, top-k 50, top-p 1; Qwen's repetition penalty 1.05; vLLM, same prompts and pixels as lmms-eval); each selector keeps one of the first N. N = 1 is the same candidate for every selector.

### All selectors at N = 16

| selector | CaptionQA score ↑ | CaptionQA agreement with own image answers ↑ | CHAIR-500 CHAIR_s ↓ | CHAIR-500 CHAIR_i ↓ | CHAIR-500 object recall ↑ | CHAIR-500 object F1 ↑ | AMBER CHAIR ↓ | AMBER Cover ↑ | AMBER Hal ↓ | AMBER Cog ↓ | words per caption (CaptionQA) | words per caption (CHAIR-500) | words per caption (AMBER) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours (S2) | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| random | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| shortest | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| self-certainty | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| Long-CLIPScore | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| Self-Validation (LPFV) | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| jina-clip-v2 (embedder) | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |

### Ours minus each selector at N = 16 (paired)

| comparison | CaptionQA score ↑ | CaptionQA agreement with own image answers ↑ | CHAIR-500 CHAIR_s ↓ | CHAIR-500 CHAIR_i ↓ | CHAIR-500 object recall ↑ | CHAIR-500 object F1 ↑ | AMBER CHAIR ↓ | AMBER Cover ↑ | AMBER Hal ↓ | AMBER Cog ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| ours − random | – | – | – | – | – | – | – | – | – | – |
| ours − shortest | – | – | – | – | – | – | – | – | – | – |
| ours − self-certainty | – | – | – | – | – | – | – | – | – | – |
| ours − Long-CLIPScore | – | – | – | – | – | – | – | – | – | – |
| ours − Self-Validation (LPFV) | – | – | – | – | – | – | – | – | – | – |
| ours − jina-clip-v2 (embedder) | – | – | – | – | – | – | – | – | – | – |

### Rank of ours among the selectors (1 = best; N = 1 is the same caption for all)

| metric | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|
| CaptionQA score ↑ | – | 4 / 7 | – | – |
| CaptionQA agreement with own image answers ↑ | – | 4 / 7 (tied with 2) | – | – |
| CHAIR-500 CHAIR_s ↓ | – | 1 / 7 (tied with 6) | – | – |
| CHAIR-500 CHAIR_i ↓ | – | 2 / 7 | – | – |
| CHAIR-500 object recall ↑ | – | 1 / 7 (tied with 4) | – | – |
| CHAIR-500 object F1 ↑ | – | 1 / 7 (tied with 3) | – | – |
| AMBER CHAIR ↓ | – | 1 / 7 | – | – |
| AMBER Cover ↑ | – | 4 / 7 (tied with 1) | – | – |
| AMBER Hal ↓ | – | 1 / 7 | – | – |
| AMBER Cog ↓ | – | 1 / 7 (tied with 6) | – | – |

## Use 2: GRPO with each reward (same config, greedy captions)

| reward | CaptionQA score ↑ | CaptionQA agreement with own image answers ↑ | CHAIR-500 CHAIR_s ↓ | CHAIR-500 CHAIR_i ↓ | CHAIR-500 object recall ↑ | CHAIR-500 object F1 ↑ | AMBER CHAIR ↓ | AMBER Cover ↑ | AMBER Hal ↓ | AMBER Cog ↓ | words per caption (CaptionQA) | words per caption (CHAIR-500) | words per caption (AMBER) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (no training) | 0.727 [0.672, 0.786] | 67.0 [58.5, 75.3] | 75.0 [25.0, 100.0] | 15.6 [2.9, 31.2] | 100.0 [100.0, 100.0] | 85.7 [76.9, 96.6] | 3.0 [0.0, 8.6] | 74.1 [62.2, 90.0] | 25.1 [0.1, 75.0] | 0.0 [0.0, 0.0] | 238.4 [202.2, 276.6] | 133.8 [124.2, 142.5] | 81.0 [69.0, 93.0] |
| ours (S2) | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| random | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| length only | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| self-certainty (Intuitor) | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| Long-CLIPScore | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| jina-clip-v2 (embedder) | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |

### Ours minus each reward (paired)

| comparison | CaptionQA score ↑ | CaptionQA agreement with own image answers ↑ | CHAIR-500 CHAIR_s ↓ | CHAIR-500 CHAIR_i ↓ | CHAIR-500 object recall ↑ | CHAIR-500 object F1 ↑ | AMBER CHAIR ↓ | AMBER Cover ↑ | AMBER Hal ↓ | AMBER Cog ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| ours − base | – | – | – | – | – | – | – | – | – | – |
| ours − random | – | – | – | – | – | – | – | – | – | – |
| ours − length only | – | – | – | – | – | – | – | – | – | – |
| ours − self-certainty (Intuitor) | – | – | – | – | – | – | – | – | – | – |
| ours − Long-CLIPScore | – | – | – | – | – | – | – | – | – | – |
| ours − jina-clip-v2 (embedder) | – | – | – | – | – | – | – | – | – | – |

### Rank of ours among the rewards (1 = best)

| metric | rank |
|---|---|

## Combination: rerank the GRPO model's samples with our reward

| setting | CaptionQA score ↑ | CaptionQA agreement with own image answers ↑ | CHAIR-500 CHAIR_s ↓ | CHAIR-500 CHAIR_i ↓ | CHAIR-500 object recall ↑ | CHAIR-500 object F1 ↑ | AMBER CHAIR ↓ | AMBER Cover ↑ | AMBER Hal ↓ | AMBER Cog ↓ | words per caption (CaptionQA) | words per caption (CHAIR-500) | words per caption (AMBER) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRPO(ours), greedy | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| GRPO(ours) + ours selector, N=1 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| GRPO(ours) + ours selector, N=2 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| GRPO(ours) + ours selector, N=4 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| GRPO(ours) + ours selector, N=8 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| GRPO(ours) + ours selector, N=16 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| base + ours selector, N=16 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |

| comparison | CaptionQA score ↑ | CaptionQA agreement with own image answers ↑ | CHAIR-500 CHAIR_s ↓ | CHAIR-500 CHAIR_i ↓ | CHAIR-500 object recall ↑ | CHAIR-500 object F1 ↑ | AMBER CHAIR ↓ | AMBER Cover ↑ | AMBER Hal ↓ | AMBER Cog ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| GRPO(ours)+ours N=16 − GRPO(ours) greedy | – | – | – | – | – | – | – | – | – | – |
| GRPO(ours)+ours N=16 − base+ours N=16 | – | – | – | – | – | – | – | – | – | – |

## Notes

- The comparison that carries the claim is within each use (selectors in Use 1, rewards in Use 2); best-of-N and GRPO are shown together but not compared with each other.
- Long-CLIP reads at most 248 tokens of a caption (longer captions are cut, as its tokenizer does); Self-Validation's LPFV checks MSCOCO objects only, so captions without such objects score 1.0 (its rule).
- CaptionQA is judged by the local text LLM (not lmms-eval's default 72B judge): absolute values are not comparable with other papers, differences between our rows are.
- AMBER Cog: lower is better (AMBER's definition; lmms-eval's task file marks it otherwise).
- Intervals: 95% percentile bootstrap over images (CaptionQA: an image's questions stay together), 1000 resamples shared by all rows, so ours − other intervals are paired.
