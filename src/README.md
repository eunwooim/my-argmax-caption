# argmax_caption_with_docci_qa · GRPO 캡셔너 + DOCCI Positive / Negative / Hallucinated QA

`argmax_caption`(Step 1 점수 고르기, Step 2 GRPO 캡셔너 학습)과 `docci_qa`(P/N/H QA 생성과 VLM 벤치마크)를 한 폴더로 합친 코드입니다.
QA의 **Caption 2는 Step 2에서 학습한 캡셔너가 씁니다.** 학습 전(LoRA 끔)과 학습 후(GRPO LoRA 켬) 두 버전이 있고, 그래서 RL 전후로 캡션의 실수(틀린 말, 빠뜨린 말)가 얼마나 바뀌었는지도 같이 봅니다.

```
run.py (Step 1: 점수 고르기) → grpo.py (Step 2: GRPO 학습, 전/후 캡션 평가, adapter/)
  → caption_docci.py (Caption 2: RL 전 / 후 × 캡션 프롬프트 simple)
  → generate.py (P/N/H QA 생성 + 자기 검사 + 정리) → benchmark.py + grade.py (VLM 평가, 채점)
  → report.py (summary.md / summary.html)
```
`python run_all.py` 한 번이면 위 전체가 순서대로 돌아갑니다.

## 설치
```bash
pip install -r requirements.txt
```

## 0) 한 번에 전부 돌리기 (`run_all.py`)

| 단계 | 스크립트 | 하는 일 | 결과 폴더 (`runs/` 아래) |
|---|---|---|---|
| A. `step1` | `run.py` | FreeRet 점수(S0~S3) 고르기, SugarCrepe·DOCCI 체크, gate | `step1/` |
| B. `grpo` | `grpo.py` | 그 점수를 보상으로 Qwen2.5-VL-7B 캡셔너 GRPO 학습(LoRA), COCO val·DOCCI에서 전/후 캡션 평가 | `grpo/` (`adapter/` 포함) |
| C. `caption` | `caption_docci.py` | DOCCI 이미지마다 Caption 2: RL 전(LoRA 끔) / 후(LoRA 켬) × 캡션 프롬프트 | `caption2_<before\|after>__<prompt>/` |
| D. `gen`, `verify` | `generate.py` | Caption 1 + Caption 2 → P/N/H QA, 만든 모델이 직접 검사, 정리(coverage) | `qa_<생성모델>__<captioner>__<prompt>/` |
| E. `bench`, `grade` | `benchmark.py`, `grade.py` | VLM이 이미지 보고(그리고 안 보고) 답함, open-ended 답 채점, 점수 | `bench_<생성모델>__<captioner>__<prompt>/` |
| F. | `report.py` | 전체 요약 | `summary.md`, `summary.html` |

```bash
python run_all.py --mock --out_root runs_mock               # 스모크 테스트: 작은 랜덤 모델·가짜 데이터, CPU, 1분 정도
python run_all.py --n_images 200 --out_root runs_try200     # QA는 DOCCI test 200장만
nohup python run_all.py > /dev/null 2>&1 &                  # 전부 (DOCCI test 5,000장), 로그는 ./runs/run_all.log
python run_all.py --step1_dir out_7b --grpo_dir out_grpo    # Step 1은 재사용. out_grpo는 128토큰으로 학습됐으니 out_grpo_tok4096에 4096으로 다시 학습
```

자주 쓰는 옵션:
- `--steps step1,grpo,caption,gen,verify,bench,grade` : 돌릴 단계 (정리, 점수, 리포트는 항상 돎)
- `--captioners before,after` : Caption 2를 쓸 캡셔너. `vllm`을 넣으면 `--caption_model`(아무 VLM)로도 만듦(`caption.py`)
- `--caption_prompts simple` : 캡션 프롬프트. Caption 2는 이제 `simple`(`Describe this image in detail.`) 하나로만 만듭니다 (detailed는 없앰)
  - `argmax`: GRPO 학습 때 쓴 `Describe this image.` (원하면 `--caption_prompts simple,argmax`)
- `--caption_max_tokens 4096`, `--caption_batch 8` : Caption 2 최대 길이, 한 번에 넣는 이미지 수
- `--generators Qwen/Qwen3-32B,openai/gpt-oss-120b` : QA 만드는 모델
- `--vlms Qwen/Qwen2.5-VL-7B-Instruct` : 벤치마크할 VLM, `--grader Qwen/Qwen3-32B` (`none`이면 단어 매칭)
- `--grpo_max_new_tokens 4096` : GRPO 학습·평가 때 캡션 최대 길이
- `--step1_args "..."`, `--grpo_args "..."` : `run.py` / `grpo.py`에 그대로 넘길 인자 (예: `--grpo_args "--steps 300 --rollouts 4"`)
- `--caption2_file my.jsonl` : 캡셔너 대신 가진 캡션 사용 (C 건너뜀)

**데이터셋 수와 계산량**: 캡셔너 2(전/후) × 프롬프트 1(simple) × 생성 모델 2 = **QA 데이터셋 4개**, Caption 2 생성은 2번입니다.
- `--generators Qwen/Qwen3-32B`면 데이터셋 2개.
- Caption 2는 GRPO와 같은 코드(HF transformers `generate`)로 만들기 때문에 vLLM보다 느립니다. 5,000장 × 최대 4096토큰을 2번이므로, 먼저 `--n_images 200`으로 시간을 재 보세요. VRAM이 남으면 `--caption_batch 16`.

**건너뛰기와 이어하기** (중간에 끊기면 같은 명령을 다시 실행):
- A: `step1/results.md`가 있으면 건너뜀.
- B: `grpo/adapter/adapter_config.json`과 `grpo/results.md`가 둘 다 있으면 건너뜀. adapter만 있으면 `--eval_only`(평가만), `ckpt/step_*`가 있으면 `--resume`(이어서 학습).
  - 단, 그 GRPO 폴더가 다른 캡션 길이로 학습됐으면(예: 예전 128토큰 `out_grpo`) 재사용하지 않습니다. `<grpo_dir>_tok4096`(예: `out_grpo_tok4096`)에 4096으로 다시 학습하고 그 adapter로 after 캡션을 만듭니다. 예전 폴더는 건드리지 않으니 128과 4096 결과를 나란히 비교할 수 있습니다.
  - 학습 길이는 `adapter/grpo_train.json`에 기록됩니다(예전 폴더는 `args.json`의 `max_new_tokens`, 없으면 128로 봄).
  - 같은 `--out_root`에 다른 adapter로 만든 after 캡션이 이미 있으면, 캡션·QA·벤치마크가 섞이지 않도록 시작할 때 멈추고 새 `--out_root`를 쓰라고 알려줍니다.
- C~E: 이미 끝난 이미지와 질문은 건너뜀.
- Step 1 gate를 통과하지 못해도 GRPO는 진행하고, 로그에 그 사실을 남깁니다(`step1/results.md` 참고).
- GRPO adapter가 없어서 after 캡션을 못 만들면 after는 건너뛰고 `summary.md`에 실패로 표시합니다. 한 단계가 실패해도 나머지는 계속 돌고, 실패한 단계와 로그 위치가 `summary.md`에 나옵니다.

**summary.md / summary.html 내용**: `run_all.py`가 만든 결과가 전부 `summary.html` 한 파일에 들어갑니다(사진 포함, 브라우저로 열면 됨). 맨 위 목차에서 각 부분으로 바로 갈 수 있습니다.
1. Step 1 결과: 점수 × 체크 표, SugarCrepe 표, gate, DOCCI 곡선 (`step1/results.md` 그대로)
2. Step 2 GRPO 결과: 전/후 점수, 승률, 캡션 길이, 예시, 학습 곡선 (`grpo/results.md` 그대로)
3. **Caption 2 길이와 잘림**: 캡셔너 × 프롬프트마다 평균 단어·토큰 수, 최대 길이에 걸려 끊긴 캡션 비율(끊긴 캡션은 뒷부분이 없어서 omission이 늘어남)
4. **Caption 2 실수 (RL 전 vs 후)**: 이미지당 positive / contradiction / omission / hallucinated 개수, hallucination이 있는 이미지 비율
5. **같은 사진, RL 전 vs 후 캡션**: 사진 몇 장에 대해 Caption 1과 모든 Caption 2(전/후 × 프롬프트)를 나란히, 캡션마다 찾은 실수 개수와 함께
6. 데이터셋 표, **QA 검사 결과**(세트별 생성 → 남은 수, 자주 버린 이유), coverage 표
7. 벤치마크 표(이미지 있음 / 없음), case별·category별 표
8. 데이터셋마다 case별(P+N+H, P+N, P+H, P, 버림) 샘플: 사진 → Caption 1 / Caption 2 → coverage → Positive / Negative / Hallucinated 표 → VLM 답과 ✅ / ⚠️ / ❌
9. 실패한 단계가 있으면 그 목록과 로그 위치

이미 돌린 결과로 리포트만 다시 만들 때 (GPU 안 씀, 몇 초):
```bash
python report.py --out_root runs --samples_per_case 4
python report.py --out_root runs --step1_dir out_7b --grpo_dir out_grpo   # A, B 결과가 다른 폴더에 있을 때
```

결과 폴더:
```
runs/
  step1/                         Step 1 (results.md, docci_curve.png, means/ ...)  임베딩 캐시는 data2로
  grpo/                          Step 2 (results.md, train_curves.png, adapter/ ...)  체크포인트는 data2로
  docci_images.jsonl             Caption 2를 만들 DOCCI 이미지 목록
  caption2_before__simple/caption2*.jsonl     RL 전 캡셔너, 간단한 프롬프트
  caption2_after__simple/
  qa_Qwen3-32B__before__simple/  qa_dataset.json (이미지별, coverage 포함)  qa_final.jsonl  stats.md
  qa_gpt-oss-120b__after__simple/ ...       (데이터셋 4개, 같은 구성)
  bench_Qwen3-32B__before__simple/<VLM>/    predictions, grades, labels, results.md, leaderboard.md
  summary.md  summary.html  samples/  run_all.log  logs/
```

## 1) Caption 2: RL 전 / 후 캡셔너 (`caption_docci.py`)
Step 2의 전/후 평가와 **같은 모델, 같은 코드**로 캡션을 만듭니다.
- 모델 로딩 `embedder.load_real`(Qwen2.5-VL-7B-Instruct, bf16, max_pixels 1024·28·28), after는 `PeftModel.from_pretrained(<grpo>/adapter)`
- 생성 `grpo.generate`: greedy, repetition penalty 1.0, 이미지 토큰 생성 금지, 채팅 형식도 grpo.py와 같음
- **before** = BeforeRlModelCaption (LoRA 끔), **after** = Argmax* (GRPO LoRA 켬)

여기서 바꾸는 것은 캡션 프롬프트뿐입니다. GRPO는 `Describe this image.`, 최대 4096토큰으로 학습·평가하고, Caption 2도 최대 4096토큰이라 길이 조건은 같습니다. 프롬프트까지 학습 때와 똑같은 캡션을 보려면 `--caption_prompts argmax`.

```bash
python caption_docci.py --which before --n_images 200 --out_dir docci_cap_before
python caption_docci.py --which after --adapter out_grpo/adapter --n_images 200 --out_dir docci_cap_after
python caption_docci.py --which after --adapter out_grpo/adapter --caption_prompt "Describe this image in detail." \
    --prompt_version simple --max_new_tokens 4096 --out_dir docci_cap_after_simple
python caption_docci.py --mock --which after --n_images 4 --out_dir docci_cap_mock     # CPU 스모크 테스트
```
`--n_images 200 --seed 0`은 grpo.py의 DOCCI 평가와 같은 200장입니다.
출력 `caption2.jsonl` 한 줄: `{"example_id", "image_file", "caption", "model", "captioner", "prompt_version", "prompt", "n_tokens", "truncated"}` (`truncated` = 최대 길이에서 잘림).

## 2) Step 1: GRPO 전에 이미지-캡션 점수 고르기 (`run.py`)
문서(Argmax_caption.docx) 1~7절 구현. frozen Qwen2.5-VL로 이미지와 캡션을 **따로** 넣어 임베딩을 하나씩 뽑고 cosine을 잽니다.
```bash
python run.py --mock --out_dir out_mock --bootstrap 2000                        # CPU 스모크 테스트 (30초 정도)
python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --out_dir out_7b
python run.py --model Qwen/Qwen2.5-VL-3B-Instruct --dtype float16 --out_dir out_3b   # T4면 3B, float16
```
단계별로 나눌 수도 있습니다: `--stage means` → `--stage select` → `--stage evaluate`. 임베딩은 `out_dir/emb/`(또는 `--emb_dir`)에 저장되어 끊겨도 이어서 돕니다. `run_all.py`로 돌리면 data2로 갑니다(아래 저장 위치).
빠르게 감만 볼 때: `--n_center 500 --limit_images 200 --bootstrap 1000` (문서 수치는 기본값으로).

출력: `results.md`(점수 × 체크 표, SugarCrepe 타입별 표, dev 선택 표, 레이어 스윕, gate), `results.json`, `docci_curve.png`, `selection.json`, `means/`, `emb/`.

문서에 없어서 정한 기본값 (모두 인자로 변경 가능):
- 레이어 번호: 1..L. S0/S1의 레이어 l = `hidden_states[l]` (L = `hidden_states[-1]`, final norm 적용됨), S2/S3의 레이어 l = l번째 레이어 post_attention_layernorm 입력. 기본은 마지막 레이어, 스윕은 선택된 점수만.
- DOCCI 색 바꾸기: 색 목록 red, orange, yellow, green, blue, purple, pink, brown, black, white, gray. 색 단어가 처음 나오는 문장의 첫 색 단어를 다른 색(이미지별 seed)으로 바꾸고, **전체 설명문** 진짜 vs 바뀐 것을 비교.
- DOCCI k문장 체크: k=1..끝까지 앞 k문장을 이어 붙여 점수, "오름" 비율 = k→k+1에서 점수가 올라간 단계 비율.
- 노이즈/오답 이미지 체크: held-out SugarCrepe 이미지마다 진짜 캡션 1개, 같은 크기 균일 노이즈 이미지 1개, 다른 held-out COCO 이미지 1개. noise·wrong 각각과 둘 다 이긴 비율을 보고.
- 동점은 진 것으로 셈 (strict `>`). 임베딩은 fp32로 저장.
- COCO는 먼저 로컬 사본 `/mnt/data2/shared_dataset/COCO-2017`(환경변수 `COCO_ROOT`로 변경)에서 `captions_<split>.json`과 `train2017/`, `val2017/`을 찾아 씁니다. 없을 때만 받음. `captions_train2017.json`이 있으면 `--coco_ann`.

## 3) Step 2: GRPO (COCO, FreeRet 보상) (`grpo.py`)
FreeRet(S2) 점수를 보상으로 Qwen2.5-VL 캡셔너를 GRPO로 학습하고, 학습 전과 후의 캡션을 같은 점수로 비교합니다.
- **BeforeRlModelCaption** = Qwen2.5-VL(이미지, 캡션 프롬프트), LoRA 끔
- **Argmax\*** = GRPO로 학습한 LoRA를 켠 Qwen2.5-VL(이미지, 캡션 프롬프트)
- **결과** = FreeRet(Argmax\*, 이미지)와 FreeRet(BeforeRlModelCaption, 이미지) 비교. COCO val2017(학습과 같은 분포)과 DOCCI test(OOD)

GPU 하나에 모델 하나만 올립니다. LoRA를 켜면 학습하는 policy, 끄면 KL 기준 모델과 보상 모델(Step 1의 S2와 똑같이 읽음) 역할을 합니다.
```bash
python grpo.py --mock --out_dir out_grpo_mock                           # CPU 스모크 테스트 (15초 정도)
python grpo.py --step1_dir out_7b --out_dir out_grpo                    # Step 1 결과(중심화 평균, 레이어)를 그대로 씀
python grpo.py --step1_dir out_7b --out_dir out_grpo --resume           # 끊긴 곳부터
python grpo.py --step1_dir out_7b --out_dir out_grpo --eval_only        # 학습 없이 평가만
```
체크포인트는 `out_dir/ckpt/`(또는 `--ckpt_dir`, `run_all.py`로 돌리면 data2). VRAM 조절: `--rollouts 4`(이미지당 캡션 수), `--batch_images 8`, `--micro_images auto`(OOM이면 절반으로 줄여 재시도), `--max_new_tokens 4096`(캡션 최대 길이. 한도일 뿐이고 실제 길이는 모델이 정함).

| 항목 | 기본값 | 이유 |
|---|---|---|
| policy 모델 | Qwen2.5-VL-7B-Instruct + LoRA r=16 (언어 모델 부분만) | Step 1과 같은 모델. 단일 GPU에서 전체 학습은 불가 |
| 보상 | Step 1에서 고른 점수 (예: S2-centered@L24) | `--step1_dir`에서 읽음 |
| 캡션 프롬프트 | `Describe this image.` | 문서에 없어서 중립적인 문장으로 정함 |
| GRPO | 이미지당 4개 샘플(temperature 1.0, repetition penalty 끔), 그룹 내 정규화 advantage, KL β=0.04, lr 1e-5, 300 step | GRPO 논문 기본값 근처 |
| 캡션 최대 길이 | `--max_new_tokens 4096` | 문서에 길이 지정이 없음. 예전 기본값 128은 근거 없이 정한 값이었고, Qwen2.5-VL 캡션이 128토큰을 자주 넘어 문장 중간에 잘렸기 때문에 4096으로 올림 |
| loss 정규화 | `--loss_norm const` (Dr. GRPO: 토큰 합 / `--loss_const 128`) | 원래 GRPO(`--loss_norm seq`)는 짧은 답을 선호하는 편향. 나누는 상수는 예전 값 128로 둬서 loss 크기와 grad clip이 걸리는 정도가 그대로 |
| 추가 평가 점수 | `--extra_scores S1,S3,S0` | 학습에 안 쓴 점수. 보상만 오르고 이것들은 그대로면 보상 해킹 신호 |
| 학습 이미지 | COCO train2017 2만 장 (Step 1 중심화 이미지 제외) | |
| 평가 | COCO val2017 500장, DOCCI test 200장, greedy | DOCCI는 Step 1과 같은 200장 |

출력: `results.md`(전/후 점수, 차이와 승률, 평균 단어 수, 예시), `eval_coco_val.jsonl`, `eval_docci_ood.jsonl`, `train_log.jsonl`, `train_curves.png`, `quick_eval.jsonl`, `adapter/`, `ckpt/step_N/`.
**주의:** Step 1에서 이 점수는 맞는 문장을 더 붙여도 오르지 않았습니다. 학습하면서 캡션이 짧아지는지 `words`를 같이 보세요.

**비교군 보상 (`--reward`, `rewards.py`)**: RL 논문의 세 런은 보상만 다르고 나머지 설정은 같습니다.
```bash
python grpo.py --step1_dir out_7b --out_dir out_grpo                                   # self: 자기 hidden-state 보상 (기본)
python grpo.py --step1_dir out_7b --out_dir out_grpo_ext --reward external             # 외부 임베딩 보상 (jina-clip-v2)
python grpo.py --step1_dir out_7b --out_dir out_grpo_len --reward length \
       --length_target_log out_grpo/train_log.jsonl                                     # length-matched control
```
- `external`: 별도 모델의 cos(이미지, 캡션). 기본 `--external_model jinaai/jina-clip-v2`(텍스트 8k 토큰, remote code의 `encode_image` / `encode_text`). CLIP / SigLIP 계열 HF 모델도 됩니다. 텍스트 한도에서 잘린 캡션 비율을 끝에 출력합니다.
- `length`: 보상 = -|단어 수 - 목표| / 목표. 목표는 self 런의 같은 step 평균 롤아웃 길이(`train_log.jsonl`의 `words`)라서, 이미지 신호 없이 길이만 self 런을 따라갑니다. self 런을 먼저 돌려야 합니다.
- 두 비교군에서도 self 점수(S2, Step 1 레이어와 평균)를 학습에 안 쓴 점수로 같이 보고합니다(`self` 행). 그래서 `--step1_dir`은 계속 필요합니다. self 런에 `--extra_external`을 주면 반대로 외부 점수를 같이 봅니다.

## 4) QA 데이터셋 생성 (`generate.py`)
텍스트 LLM(기본 **Qwen3-32B**, 그리고 **gpt-oss-120b**)이 Caption 1(DOCCI 사람 설명, 정답)과 Caption 2로 세 종류의 QA를 만듭니다. 프롬프트 원문은 **`pnh_prompt.md`**에 있습니다(여기서 고치면 바로 반영).

| 세트 | 정의 | 정답 |
|---|---|---|
| **Positive** | 두 캡션 모두로 풀 수 있는 질문 (같은 답) | 두 캡션의 공통 답 |
| **Negative** | Caption 1으로 풀 수 있는데 Caption 2가 반대로 말함(`contradiction`) 또는 빠뜨림(`omission`) | Caption 1의 답 |
| **Hallucinated** | Caption 2가 Caption 1과 충돌하게 지어낸 내용 | Caption 2의 틀린 주장, `caption1_truth`에 Caption 1의 답 |

공통 규칙(프롬프트): 질문 하나에 사실 하나, "According to Caption 1" 같은 메타 표현 금지, 답은 1~5단어, `category`는 object/attribute/action/count/spatial/scene/text/relational, `format`은 open-ended 또는 mcq(A~D). 생성 temperature 0.2.

| 단계 | 하는 일 | 출력 |
|---|---|---|
| `gen` | 이미지마다 두 캡션을 `pnh_prompt.md`로 넣고 P/N/H JSON 생성 (JSON 스키마 강제) | `gen_raw.jsonl` |
| `verify` | 만든 모델이 직접 판정: 그 세트가 맞는지(`valid`), 사실 하나만 묻는지(`atomic`). 다른 모델로 교차 검사는 안 함 | `verify/<모델>.jsonl` |
| `build` | 규칙 검사 + 판정 + 이미지별 coverage | `qa_dataset.json`, `qa_final.jsonl`, `stats.md` |

```bash
python generate.py --caption2 runs/caption2_after__simple/caption2.jsonl --out_dir out_qa_after_simple
python generate.py --caption2 cap2/caption2.jsonl --n_images 200 --out_dir out_qa_200
python generate.py --caption2 cap2/caption2.jsonl --qa_format mcq --out_dir out_qa_mcq    # mixed도 가능
python generate.py --model openai/gpt-oss-120b --caption2 cap2/caption2.jsonl --out_dir out_qa_gptoss
```
옵션: `--qa_format open-ended|mcq|mixed`, `--hal_mcq_truth`(hallucinated MCQ에 Caption 1 정답도 보기로), `--max_answer_words 5`, `--keep_meta`, `--keep_no_positive`, `--prompt 다른_프롬프트.md`.

**정보가 부족한 이미지 (coverage)**: positive가 하나 이상이면 남기고, 이미지마다 어떤 세트가 있는지 적습니다.

| case | 뜻 |
|---|---|
| `P+N+H` | 세 세트 모두 |
| `P+N` | positive + negative (Caption 2가 틀린 말은 안 하고 빠뜨리기만 함) |
| `P+H` | positive + hallucinated |
| `P` | positive만 (두 캡션이 다 일치) |
| positive 없음 | 빠지고 `dropped_images`에 이유와 함께 기록 (`no_positive`, `json_parse_failure`, `no_caption2`) |

`qa_dataset.json`의 이미지 하나:
```json
{"example_id": "test_00331", "split": "test", "image_file": "test_00331.jpg",
 "caption1": "...", "caption2": "...", "caption2_model": "Qwen/Qwen2.5-VL-7B-Instruct + GRPO LoRA (runs/grpo/adapter)",
 "caption2_prompt": "simple", "gen_model": "Qwen/Qwen3-32B",
 "coverage": {"case": "P+N", "complete": false, "positive": true, "negative": true, "hallucinated": false,
              "n": {"positive": 4, "negative": 2, "hallucinated": 0},
              "n_generated": {"positive": 5, "negative": 2, "hallucinated": 1},
              "neg_types": {"contradiction": 0, "omission": 2}},
 "positive": [{"id": "pos_1", "qa_id": "test_00331_pos0", "question": "Where is the man located relative to the tree?",
               "answer": "in front of a tree", "category": "spatial", "format": "open-ended"}],
 "negative": [{"id": "neg_1", "qa_id": "test_00331_neg0", "question": "What color is the jump rope?", "answer": "red",
               "neg_type": "omission", "category": "attribute", "format": "open-ended"}],
 "hallucinated": []}
```
- `n_generated`는 검사 전, `n`은 검사 후 개수. 위 예시는 hallucinated 1개가 검사에서 떨어져 `P+N`이 된 경우입니다.
- 같은 사실의 contradiction negative와 hallucinated는 `pair_qa_id`로 연결되고, negative에는 Caption 2가 한 말이 `caption2_claim`으로 붙습니다.
- MCQ면 `options`(A~D), `answer`(정답 글자), `answer_text`가 들어갑니다.

**규칙 검사 (build)**: 질문/답이 비었거나 물음표로 안 끝남, 메타 표현(caption, according to …)이 남음, category/format/neg_type이 정해진 값이 아님, 답이 `--max_answer_words`보다 김, MCQ 보기가 4개가 아니거나 겹치거나 정답 글자가 없음, hallucinated에서 claim과 truth가 같음, 같은 세트 안 중복이면 버립니다. 판정은 `valid`와 `atomic`이 모두 true여야 통과. 버린 이유는 `stats.md`에.
프롬프트의 few-shot 예시에 있는 "according to the candidate claim"은 Context-Free 규칙과 충돌해서 build에서 지웁니다(원래 질문은 `question_raw`, 안 지우려면 `--keep_meta`).

**gpt-oss-120b**: `--model`에 gpt-oss 이름을 주면 자동으로 맞춰집니다(추론 후 final만 JSON으로 읽음, `--reasoning_effort low|medium|high`, 스키마 강제 대신 파싱 + 재시도, max_tokens 6144). GPU 1장에 올라갑니다(MXFP4, vLLM 0.10.1 이상).
**이미 띄운 서버**: `--backend openai --base_url http://localhost:8000/v1` (vLLM / SGLang 등 OpenAI 호환 서버).

## 5) 벤치마크 (`benchmark.py`, `grade.py`)
VLM에 이미지 + 질문만 넣습니다(캡션은 안 보여줌). open-ended는 "Answer in a few words.", MCQ는 보기 4개와 "Answer with the letter". 같은 이미지의 같은 질문은 한 번만 묻습니다.

답마다 라벨: `correct`(참인 답), `false_claim`(Caption 2의 틀린 주장을 따라 함), `wrong`, `invalid`(MCQ에서 글자를 못 찾음). open-ended 답은 정확히 일치하면 바로, 아니면 `--grader`(기본 Qwen3-32B)가 판정합니다.
지표(이미지 단위 부트스트랩 95% 구간): positive 정확도, negative 정확도(contradiction / omission 따로), hallucinated **truth 정확도**와 **repeat 비율 ↓**, contradiction repeat 비율 ↓, case별, category별.
```bash
python benchmark.py --qa out_qa/qa_final.jsonl --model Qwen/Qwen2.5-VL-7B-Instruct --no_score --out_dir out_bench
python benchmark.py --qa out_qa/qa_final.jsonl --model Qwen/Qwen2.5-VL-7B-Instruct --no_image --no_score --out_dir out_bench
python grade.py --runs 'out_bench/*'
python benchmark.py --qa out_qa/qa_final.jsonl --model Qwen/Qwen2.5-VL-7B-Instruct --score_only --out_dir out_bench
```
주의: 캡셔너와 같은 모델(Qwen2.5-VL-7B)을 벤치마크하면 negative/hallucinated가 그 모델 자신의 실수로 만들어진 것이라 점수가 낮게 나오는 게 정상입니다.

## 6) 다른 VLM으로 Caption 2 (`caption.py`)
GRPO 캡셔너가 아닌 아무 VLM(vLLM)으로 Caption 2를 만들 때. `run_all.py --captioners before,after,vllm --caption_model Qwen/Qwen3-VL-8B-Instruct`처럼 같이 돌릴 수도 있습니다.
```bash
python caption.py --model Qwen/Qwen3-VL-8B-Instruct --prompt_version simple --out_dir cap2_qwen3vl
```
greedy, 최대 4096토큰(`--max_tokens`), repetition penalty 1.05(`--repetition_penalty 1.0`이면 끔).


## 7) 짧은 캡션 ablation (`ablation_length.py`)
"GRPO 후 보상이 오른 게 캡션이 짧아져서인가?"를 확인합니다. grpo.py가 평가한 같은 이미지(`eval_coco_val.jsonl`, `eval_docci_ood.jsonl`)에서, 학습 보상(S2-centered@L24, LoRA 끔)으로 아래 캡션들을 채점합니다.
- `before` / `after`: RL 전 / 후 캡션
- `before_lenmatch`: before를 문장 단위로 after 길이만큼 자름, `before_words`: 단어 수를 정확히 맞춰 자름
- `before_k1/k2/k3`: before의 첫 1/2/3 문장
- `short_1`, `short_2`: RL 전 모델에게 짧게 쓰라고 시킨 캡션 (`Describe this image in one sentence.`, `Briefly describe this image.`; `--short_prompts`로 변경)

`after − row`가 0보다 확실히 크면(구간 하한 > 0) "짧기만 해서는 점수가 안 오른다"가 맞습니다. before 캡션 안에서 길이와 점수의 상관도 같이 적습니다.
```bash
python ablation_length.py --step1_dir out_7b --grpo_dir out_grpo_tok4096      # GPU 1개 (0, 1, 2, 3 제외)
python ablation_length.py --mock --grpo_dir runs_mock/grpo                      # CPU 스모크 테스트
```
결과: `<grpo_dir>/ablation_length/results.md`, `results.html`, 이미지별 `per_image_<set>.jsonl`.

## 8) 실제 결과 샘플 HTML (`sample_html.py`)
QA 폴더에서 무작위 이미지 N장을 골라 사진, Caption 1(사람), Caption 2(생성된 캡션), coverage, Positive / Negative / Hallucinated 질문과 정답, VLM의 답 전체(이미지 있음 / `__noimage`)와 채점을 한 HTML에 보여줍니다. GPU 필요 없음.
```bash
python sample_html.py --out_root runs                         # 기본: '*__before__simple' (RL 전 × simple, 생성 모델마다), coverage case별 P+N+H 20 · P+N 10 · P+H 10 · P 10장
python sample_html.py --out_root runs --per_case "P+N+H=5,P=5"   # case별 장수 변경
python sample_html.py --out_root runs --per_case "" --n 100     # case 구분 없이 무작위 100장
python sample_html.py --out_root runs --case P+N+H            # 특정 coverage case만
python sample_html.py --out_root runs --datasets '*__after__simple' --out runs/sample_after_simple.html
```
결과: `runs/sample_before_simple.html` (사진 포함 파일 하나).

## 9) RL 논문: GRPO 체크포인트를 lmms-eval로 평가 (`lmms_eval_run.py`, `lmms_eval_lora.py`, `lmms_eval_report.py`)
고정한 체크포인트(base, self, external, length)를 표준 벤치마크로 같은 설정에서 평가합니다. mcq·general 계열은 "다른 능력이 망가지지 않았다", caption 계열(COCO CHAIR, DOCCI 캡셔닝)은 "캡션이 실제로 좋아졌다"를 보여주는 쪽입니다. (2026-11-06: 성능 차이가 없거나 judge 의존이 큰 AMBER-G, CaptionQA, DetailCaps, LLaVA-Bench, MMHal-Bench, POPE A-OKVQA는 뺐고 DOCCI 캡셔닝을 넣었습니다.) 논문 주 지표(캡션 vs 같은 모델의 시각 답 일치도, DOCCI QA)는 지금처럼 `caption_docci.py --adapter <런>/adapter` → `run_all.py` 경로로 계속 잽니다.

**설치** (lmms-eval 0.7.4, commit 5d62685에서 확인):
```bash
pip install lmms-eval qwen-vl-utils           # 또는 git clone 후 pip install -e
pip install capture_metric pycocoevalcap pycocotools nltk openpyxl
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet omw-1.4   # CHAIR
java -version                                  # 캡션 지표(METEOR/PTB tokenizer)용. 없으면 conda install -c conda-forge openjdk=11
python lmms_eval_smoke.py --out_dir lmms_smoke  # CPU 1분: mock 학습 → merge 검사 → 실제 lmms-eval(어댑터 메모리 merge) → 표. 먼저 한 번
```

**1. 체크포인트 병합 검사**: grpo.py는 LoRA 어댑터만 저장하고, lmms-eval의 `qwen2_5_vl`은 `pretrained=<폴더>` 하나만 읽습니다. 어댑터를 base에 merge한 7B 폴더(약 16 GB)를 따로 저장하지 않고, `lmms_eval_run.py`가 어댑터 폴더를 받으면 `lmms_eval_lora.py`를 통해 lmms-eval이 base를 읽은 직후 메모리에서 merge합니다(`export_ckpt.py`가 저장 전에 하는 것과 같은 단계라 가중치도 같음. 스모크 테스트가 문항별 출력이 같은지 확인). 평가 전에 merge가 맞는지만 확인합니다(디스크 거의 0):
```bash
python export_ckpt.py --check_only --adapter out_grpo/adapter --grpo_dir out_grpo --out merge_checks/self
```
확인하는 것(하나라도 틀리면 멈춤, 결과는 `export.json`): ① merge한 모델이 base + LoRA와 얼마나 다른지(greedy 캡션 전체 위치에서 teacher-forced로 비교: top-1 일치율 ≥ 0.95, 평균 KL이 LoRA가 바꾼 양(base + LoRA vs base)의 20% 이하이거나 0.01 이하. bf16 merge는 W + BA를 bf16으로 반올림하므로 조금은 달라지는 게 정상), ③ processor의 chat template이 학습 프롬프트와 글자 단위로 같은지. `--check_only` 없이 돌리면 merge한 HF 폴더를 저장하고 ② 다시 읽었을 때 메모리의 merge 모델과 logit이 같은지도 봅니다(다른 곳에 모델을 넘길 때).

**2. 실행**: 모델 × 태스크를 같은 설정으로 돌립니다. 처음 실행할 때 `plan.json`에 태스크 목록과 설정을 고정하고, 이후 다른 태스크나 설정으로 돌리면 멈춥니다(결과 보고 나서 목록 바꾸기 방지). 모델 이름과 새 태스크 그룹 추가는 됩니다(추가한 시각이 `plan.json`에 남고, 기존 그룹은 그대로). 그룹을 빼려면 `--drop_groups oeq,selfaug`처럼 명시해야 하고, 뺀 그룹의 결과는 `_old_<시각>/`에 남습니다(표에서만 빠짐). 같은 실행에서 새로 넣은 그룹에 뺀 그룹과 같은 태스크가 같은 생성 설정으로 있으면 그 칸의 응답 캐시를 복사해서 생성 없이 채점만 다시 합니다. 전부 바꿀 땐 `--replan`(이전 계획과 결과는 따로 보관).
```bash
python prepare_tasks.py --out_dir custom_tasks --n_images 1000                  # DOCCI 캡셔닝 태스크 (한 번; 0 = test 5,000장 전부)
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000 --gpu-memory-utilization 0.4 --max-model-len 8192 &    # 채점용 LLM (약 30 GB, GPU 1장의 40%) (MM-Vet)
python lmms_eval_run.py --out_dir lmms_out --gpus 4 --hf_dir /mnt/data2/<me>/lmms_eval_work/hf --judge_url http://localhost:8000/v1 --judge_model Qwen/Qwen2.5-14B-Instruct --include_path custom_tasks \
  --models base=Qwen/Qwen2.5-VL-7B-Instruct,self=out_grpo/adapter,external=out_grpo_external/adapter,length=out_grpo_length/adapter
python lmms_eval_run.py --out_dir lmms_out --dry_run     # 고정된 계획의 명령만 출력
python lmms_eval_report.py --out_dir lmms_out           # summary.md / summary.html: 모델 × 지표, base 대비 차이, 캡션 평균 단어 수, 같은 이미지의 모델별 캡션
```

| 그룹 | 태스크 (lmms-eval 이름) | 생성 설정 |
|---|---|---|
| mcq | `pope`, `mmstar` | 각 태스크 기본값 |
| caption | `coco_cap_chair`, `docci_caption` | 전부 greedy, 최대 토큰 = 학습 한도(128, `out_grpo/args.json`) |
| general | `mmvp`, `mmbench_en_dev_static`, `mme`, `mmvet` | 각 태스크 기본값, 단 greedy |

**docci_caption** (`prepare_tasks.py`가 태스크 폴더로 만듭니다): DOCCI test 이미지(GRPO는 COCO train2017로만 학습했고 DOCCI test는 QA·전후 평가에만 쓴 held-out)에 학습 프롬프트 "Describe this image."로 캡션 하나를 만들고 사람이 쓴 DOCCI 설명(평균 130단어대)과 비교합니다. 지표는 CAPTURE(DetailCaps의 상세 캡션 지표: 캡션과 참조의 객체·속성·관계 일치, `capture_metric`, GPU 사용), CIDEr, BLEU-4, METEOR, ROUGE-L(`pycocoevalcap`, Java tokenizer), 그리고 표의 캡션 평균 단어 수입니다. 기본은 test 5,000장 중 seed 0으로 고른 1,000장(이미지 약 250 MB, QA 파이프라인의 DOCCI 캐시에 저장)이고 `--n_images 0`이면 전부입니다. 참조가 한 장에 하나라 CIDEr·BLEU 절대값은 낮게 나오며, 모델끼리 비교만 합니다.

**general 그룹**: MMVP, MMBench(`mmbench_en_dev_static`, 답 추출을 규칙으로 해서 API 키 불필요), MME(Perception + Cognition), MM-Vet. MM-Vet만 로컬 judge(Qwen2.5-14B)가 채점하므로 절대 점수는 논문들과 비교할 수 없고 우리 모델끼리만 비교합니다. MM-Vet 기본값 temperature 0.2는 greedy로 바꿨습니다. judge가 죽어 있으면 그 칸은 시작하지 않고 기다리며, 채점 결과가 비어 있으면(MM-Vet 0점) 완료로 치지 않고 따로 치워 둡니다.

- 이미지 크기는 학습과 같게(`min_pixels=3136`, `max_pixels=802816`, 학습 설정에서 읽음) 넘깁니다. lmms-eval 기본값(256·28·28 ~ 1605632)과 다릅니다.
- `--force_simple` 경로를 씁니다. lmms-eval의 chat 경로는 이미지 크기 한도를 이미지에 넘기지 않아서입니다. 같은 프롬프트, 같은 greedy로 lmms-eval이 낸 캡션이 grpo.py 생성 코드의 캡션과 같은 것을 mock에서 확인했습니다.
- caption 그룹은 태스크 기본값 대신 greedy와 학습 한도를 씁니다(태스크 기본 토큰 한도가 긴 캡션을 자르지 않게). 논문에 이 차이를 적어야 합니다. 현재 `plan.json`의 한도는 128입니다.
- **속도**: `--batch_size 8`(기본)로 프롬프트 8개를 한 번에 생성하고(left padding, greedy), `--attn auto`는 flash-attn이 설치돼 있으면 flash attention을 씁니다(학습·export도 `ATTN_IMPL` 환경변수로 같은 선택). 둘 다 plan에 고정되므로 한 표 안의 모든 칸이 같은 커널·batch로 돌아갑니다(bf16에서는 커널과 batch에 따라 출력이 아주 조금 달라질 수 있어서, 섞지 않습니다). 2026-11-03 이전 실행은 batch 1, sdpa였습니다.
- **공용 서버와 디스크**: 공용 HF 캐시(`HF_HOME`)에는 쓰지 않고 읽기만 합니다(거기 있는 Qwen2.5-VL-7B는 그대로 씀). 모든 다운로드는 `--hf_dir` 아래로 갑니다: `models/`(캐시에 없는 모델, 유지), `datasets_<그룹>/`(벤치마크 데이터 + arrow 캐시 + xet 캐시). 그룹 단위로 돌아서(모든 모델 × 한 그룹, 다음 그룹) 한 그룹이 모든 모델에서 끝나면 그 데이터 폴더를 지웁니다. 그래서 디스크에는 한 그룹의 데이터(최대 약 10 GB)만 올라가 있습니다(`--keep_datasets`로 유지, `--disk`로 현재 사용량 출력). xet 다운로드는 끕니다(`HF_HUB_DISABLE_XET=1`, 네트워크 디스크에서 writer 에러). 첫 실패에서 바로 멈추고 run.log의 에러 부분을 보여 줍니다(`--keep_going`으로 계속). 결과 파일이 없으면 lmms-eval이 exit 0이어도 실패로 칩니다.
- 시작 전에 확인하고 멈춥니다: 채점 서버 응답, `docci_caption` 태스크 폴더, 각 태스크 채점 모듈 설치 여부, Java(METEOR), CAPTURE import, NLTK 데이터(CHAIR), `--hf_dir` 디스크 여유(10 GB 미만이면 멈춤).
- lmms-eval은 이미지를 JPEG(품질 85)로 다시 인코딩해서 넣습니다. 모든 모델에 똑같이 적용되니 비교에는 영향이 없지만, 학습 때 입력과 픽셀 단위로 같지는 않습니다.

### 외부 프롬프팅 베이스라인 (`prompting_baselines.py`, BASELINES.md)

base 모델에 다른 논문의 training-free 방법(Self-Validation BoN/FtA, arXiv 2601.22451; LogicCheckGPT, arXiv 2402.11622)을 얹어 같은 태스크·설정으로 평가합니다. 공식 코드 기본값, 프롬프트는 원문 그대로이고 바꾼 부분은 전부 [BASELINES.md](BASELINES.md)에 있습니다. `bash rl_paper_commands.sh baselines_quick` → `baselines 3` → `report_all`(기존 표에 베이스라인 열 추가). CPU 점검: `python prompting_smoke.py`.

`rl_paper_commands.sh`의 judge는 `JUDGE_PORT`(8047)에 띄웁니다. 서버의 8000번 포트는 다른 사용자의 것이므로 쓰지 않습니다.

## 10) Reward comparison: test-time scaling + GRPO (`tts.py`, `tts_report.py`, `prepare_tts_tasks.py`)

제안한 reward(모델 자신의 image-only vs caption-only hidden-state 유사도, S2)가 좋은 신호인지 두 용도로 시험합니다(Math-Shepherd, arXiv 2312.08935와 같은 틀). 주장은 **각 용도 안에서 reward끼리 비교**한 순위이고, best-of-N과 GRPO를 서로 비교하지는 않습니다.

| 용도 | 무엇을 | 비교 대상 |
|---|---|---|
| Use 1: test-time scaling | base 모델이 이미지당 16개 샘플(Self-Validation 공식 샘플링: temperature 0.5, top-k 50, top-p 1), selector가 처음 N개(1, 2, 4, 8, 16) 중 하나를 고름 | ours(S2), random, shortest, self-certainty(arXiv 2502.18581), Long-CLIPScore, Self-Validation LPFV, jina-clip-v2 |
| Use 2: GRPO | 같은 GRPO 설정(128-token rollout, 300 step, seed 0)에 reward만 바꿈 | ours(`out_grpo`), random, length(`out_grpo_length`), self-certainty(Intuitor, arXiv 2505.19590), Long-CLIP, jina-clip-v2(`out_grpo_external`) |
| Combination | self-reward GRPO 모델의 16개 샘플을 ours로 다시 고름 | GRPO greedy, base + ours |

- **벤치마크(두 용도 공통, `caption_long` 그룹):** CaptionQA(점수 + 캡션만 본 답과 모델 자신이 이미지를 보고 낸 답의 일치도), Self-Validation의 CHAIR-500(CHAIR_s / CHAIR_i / recall / F1, 공식 `chair.py` + `chair.pkl`), AMBER, 캡션 길이. 캡션 길이 한도는 없습니다(4096). 128로 학습한 adapter도 평가는 4096입니다.
- **순서:** `rc_setup`(다운로드) → `rc_train` → `judge` → `rc_quick`(4개씩 전 단계 점검) → `rc_eval 2` / `tts_sample 2` → `tts_score` → `tts_select` → `tts_vqa` → `tts_eval 3` → `rc_report`(`tts_out/reward_comparison.md`, `.html`). CPU 점검: `python tts_smoke.py --selfval_dir <SelfVal clone>`.
- 샘플링은 lmms-eval 안에서 합니다(`prompting_baselines.py`의 `tts_sample`). 그래서 프롬프트와 이미지가 벤치마크와 같고, 모델이 본 JPEG 바이트를 그대로 저장합니다. 점수 계산과 선택은 `tts.py`가 오프라인으로 하고, 고른 캡션은 `replay` 칸이 lmms-eval 채점기에 그대로 넣습니다. 그래서 Use 1과 Use 2의 채점이 완전히 같습니다.
- 모든 숫자에 이미지 단위 95% bootstrap 구간이 붙습니다. "ours − 다른 것"은 같은 이미지, 같은 resample로 짝지어 계산합니다.

## 저장 위치
- 결과와 로그는 **실행한 폴더**에 생깁니다: `runs/`(`--out_root`로 변경). 가장 큰 것이 GRPO adapter(약 0.2 GB)이고 나머지는 작습니다.
- 큰 파일은 모두 `/mnt/data2/mchung45` 아래로 갑니다.
```
./runs/                                           결과, 단계별 로그, run_all.log, summary.md/html, grpo/adapter
/mnt/data2/mchung45/argmax_caption_work/outputs/<결과 폴더 경로>/
  emb/           Step 1 임베딩 캐시 (수 GB)          --big_root로 변경, 환경변수 ARGMAX_WORK_DIR
  ckpt/step_N/   GRPO 체크포인트 (하나에 약 0.5 GB)
/mnt/data2/mchung45/docci_qa_work/              QA 쪽 (qa_common.py의 WORK_ROOT, 환경변수 DOCCI_QA_ROOT)
  cache/docci/   DOCCI 설명과 이미지 (1~2 GB). run_all.py가 test 전체를 받으면 Step 1 / GRPO도 여기서 읽음
  caches/        vLLM·torch 컴파일 캐시, 임시 파일(TMPDIR), .pyc
/mnt/data2/mchung45/argmax_caption_work/cache/  Step 1 / GRPO 쪽 다운로드 (SugarCrepe, 로컬에 없을 때의 COCO; 환경변수 ARGMAX_CACHE_DIR)
```
COCO는 로컬 `/mnt/data2/shared_dataset/COCO-2017`을 씁니다. 모델 가중치는 HF_HOME이 이미 설정돼 있으면(공유 모델 폴더 `/mnt/shared/shared_hf_home`) 그대로 쓰고, 없으면 `docci_qa_work/caches/hf`로 받습니다.
예전에 돌린 폴더에 `emb/`나 `ckpt/`가 이미 있으면(예: `--step1_dir out_7b --grpo_dir out_grpo`) 그 자리를 그대로 이어서 씁니다.
시작할 때 각 위치의 남은 용량을 확인하고, 모자라면 다운로드나 학습 전에 멈춥니다.

## GPU (H200 기준)
- **GPU 0, 1, 2, 3은 절대 쓰지 않습니다**(`gpu.py`의 `FORBIDDEN`, `qa_common.py`의 `FORBIDDEN_GPUS`). `--gpus`에 넣어도 무시. 기본(`--gpus auto`)은 그 외 모든 GPU 중 빈 GPU.
- A, B(Step 1, GRPO): 빈 GPU 1장(원래 run.py / grpo.py와 같음).
- C, D, E: 빈 GPU마다 모델 하나씩 올리고 이미지를 나눠 동시에 처리(데이터 병렬). 빈 GPU가 5장이면 5배 정도 빠름. `--gpus_per_model 2`(텐서 병렬), `--max_parallel 4`.
- 빈 GPU가 없으면 생길 때까지 기다립니다(`--no_gpu_wait`로 끔). 샤드별 로그는 `runs/logs/`, 2분마다 각 샤드의 마지막 줄을 화면에 찍습니다.

## 디스크
시작할 때 모델·DOCCI·결과·임시 폴더의 남은 용량을 출력하고, 모자라면 다운로드 전에 멈춥니다(`--skip_disk_check`로 무시). 캐시와 임시 파일 위치는 `--cache_root`.

## 파일
| 파일 | 내용 |
|---|---|
| `run_all.py` | 전체 파이프라인 (Step 1 → GRPO → Caption 2 → QA 생성·검사·정리 → 벤치마크 → 채점 → summary) |
| **캡셔너 (argmax_caption)** | |
| `run.py` | Step 1 (means → select → evaluate) |
| `grpo.py` | Step 2 GRPO 학습 + 전/후 캡션 평가 |
| `caption_docci.py` | 그 캡셔너(전/후)로 DOCCI Caption 2 |
| `rewards.py` | 비교군 보상: 외부 임베딩(`ExternalReward`), length-matched control(`LengthReward`) |
| `export_ckpt.py` | LoRA merge 검사 (`--check_only`), 또는 merge된 HF 폴더 저장 + 동일성 검사 |
| `lmms_eval_lora.py` | lmms-eval 실행 시 LoRA 어댑터를 base에 메모리에서 merge (모델 폴더 저장 없음) |
| `lmms_eval_run.py` | 모델 × 태스크 lmms-eval 실행, 계획 고정, 사전 점검 |
| `lmms_eval_report.py` | lmms-eval 결과 표 (summary.md / summary.html / summary.json) |
| `prompting_report.py` | self 우위 벤치마크 3분류에서 체크포인트와 프롬프팅 베이스라인 비교 (md / html) |
| `lmms_eval_smoke.py` | lmms-eval 경로 전체 CPU 스모크 테스트 (DOCCI 캡셔닝 태스크, 그룹 drop/추가, rescore 포함) |
| `prepare_tasks.py` | lmms-eval에 없는 DOCCI 캡셔닝 태스크(`docci_caption`) 만들기 |
| `rl_paper_commands.sh` | RL 논문 전체 단계 명령 (train → verify → judge → par/eval → report, purge/clean) |
| `prompts.py` | S0~S3 점수 프롬프트 (FreeRet 규칙 문장은 원본 `input_format_oneword.py`에서 복사) |
| `embedder.py` | 모델 로딩, 임베딩 읽는 위치, 풀링, mock 모델 |
| `data.py` | SugarCrepe, DOCCI(test 200장), COCO |
| `checks.py` | 체크 1~4, 부트스트랩 구간, gate |
| `gpu.py` | 빈 GPU 고르기 (0, 1, 2, 3 제외) |
| `mock_data.py` | 스모크 테스트용 합성 데이터 (Step 1, GRPO) |
| **QA (docci_qa)** | |
| `pnh_prompt.md` | QA 생성 프롬프트 원문 (system + user) |
| `generate.py` | P/N/H 생성 → 검사 → 정리, coverage |
| `benchmark.py` | VLM 평가, 이미지 없는 기준선, 라벨과 점수, 리더보드 |
| `grade.py` | open-ended 답 판정 (LLM) |
| `report.py` | summary.md / summary.html (Step 1, GRPO 결과 포함) |
| `caption.py` | 아무 VLM(vLLM)으로 Caption 2 |
| `ablation_length.py` | 짧은 캡션 ablation: 길이만 줄여도 보상이 오르는지 |
| `sample_html.py` | 실제 결과 샘플 HTML (기본: RL 전 × simple) |
| `qa_prompts.py` | 캡션 프롬프트(simple / argmax), 판정·평가 프롬프트, JSON 스키마 (docci_qa의 `prompts.py`) |
| `qa_common.py` | DOCCI 로딩, GPU 선택, 저장 위치, 샤드, 부트스트랩 (docci_qa의 `common.py`) |
| `llm.py` | vLLM 직접 로딩, OpenAI 호환 서버, mock |
| `mock_docci.py` | 스모크 테스트용 합성 이미지와 캡션 (RL 전/후 캡셔너 흉내 포함) |

두 코드의 `prompts.py`가 이름이 겹쳐서 QA 쪽은 `qa_prompts.py`, `common.py`는 `qa_common.py`로 이름만 바꿨습니다. 내용은 같습니다.

**mock 주의**: `--mock`에서 Step 1과 GRPO는 작은 랜덤 Qwen2.5-VL로 진짜 코드 경로를 돌지만 그 캡션은 의미 없는 글자라서, QA 단계의 Caption 2는 `mock_docci.py`의 가짜 캡션(RL 전/후 차이를 흉내 냄)을 씁니다. `caption_docci.py --mock`은 따로 돌려서 확인할 수 있습니다.
