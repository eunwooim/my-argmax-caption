# Step 2: GRPO with FreeRet reward

Reward: **S2-raw@L4** (frozen; LoRA off). Policy: mock + LoRA r=16. Caption prompt: `Describe this image.`. Rollouts 4, batch 2 images, 3 steps, lr 1e-05, KL beta 0.04, loss norm const.

Before = BeforeRlModelCaption (base model). After = Argmax* (GRPO-trained). Greedy decoding for both. Intervals: 95% bootstrap over images.

`reward` is the score that was optimised, so it should go up by construction. The other rows were never trained on: if they do not move with it, the gain is specific to the reward (reward hacking), not better captions.

| set | scorer | before | after | after - before | after wins |
|---|---|---|---|---|---|
| coco_val (n=4) | reward (S2-raw@L4) | 0.8468 | 0.8468 | +0.0000 [+0.0000, +0.0000] | 0.0% [0.0, 0.0] |
| coco_val (n=4) | S1 (S1-raw@L4) | 0.3826 | 0.3826 | +0.0000 [+0.0000, +0.0000] | 0.0% [0.0, 0.0] |
| coco_val (n=4) | S3 (S3-raw@L4) | 0.9169 | 0.9169 | +0.0000 [+0.0000, +0.0000] | 0.0% [0.0, 0.0] |
| coco_val (n=4) | S0 (S0-raw@L4) | 0.3757 | 0.3757 | +0.0000 [+0.0000, +0.0000] | 0.0% [0.0, 0.0] |
| docci_ood (n=3) | reward (S2-raw@L4) | 0.8716 | 0.8692 | -0.0024 [-0.0073, +0.0000] | 0.0% [0.0, 0.0] |
| docci_ood (n=3) | S1 (S1-raw@L4) | 0.4771 | 0.4666 | -0.0105 [-0.0315, +0.0000] | 0.0% [0.0, 0.0] |
| docci_ood (n=3) | S3 (S3-raw@L4) | 0.9322 | 0.9313 | -0.0009 [-0.0027, +0.0000] | 0.0% [0.0, 0.0] |
| docci_ood (n=3) | S0 (S0-raw@L4) | 0.4736 | 0.4621 | -0.0115 [-0.0346, +0.0000] | 0.0% [0.0, 0.0] |

| set | words before | words after |
|---|---|---|
| coco_val | 1.2 | 1.2 |
| docci_ood | 1.0 | 1.0 |

Step 1 found this score does not rise with more true detail, so shrinking captions are a warning sign.

## Examples (coco_val)

- **['coco', 5000]** before (0.852): �7��kr��
  after (0.852): �7��kr��
- **['coco', 5001]** before (0.828): �7��kÖ�
  after (0.828): �7��kÖ�
- **['coco', 5002]** before (0.827): �7�����
  after (0.827): �7�����
- **['coco', 5003]** before (0.881): �)N)N� )N
  after (0.881): �)N)N� )N

## Examples (docci_ood)

- **['docci', 'synth_1000']** before (0.903): �7���L���
  after (0.903): �7���L���
- **['docci', 'synth_1001']** before (0.926): ��L����L�
  after (0.926): ��L����L�
- **['docci', 'synth_1002']** before (0.786): �7�����
  after (0.779): �7����

