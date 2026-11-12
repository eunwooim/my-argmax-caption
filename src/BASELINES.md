# 외부 프롬프팅 베이스라인 (training-free)

GRPO 체크포인트가 base보다 나은 벤치마크(MMStar, MME, MMBench, COCO CHAIR/Recall, MM-Vet, DOCCI 캡셔닝)에 다른 논문의 training-free 방법을 **base 모델(Qwen2.5-VL-7B-Instruct)에** 적용합니다. 코드는 `prompting_baselines.py`이고, lmms-eval의 `qwen2_5_vl` 모델 위에 얹습니다(`generate_until`만 감쌈). 그래서 프로세서, chat template, 이미지 크기(min/max pixels), 태스크 프롬프트, 채점기, 생성 길이는 체크포인트를 평가할 때와 완전히 같습니다. 설정은 `--settings_from lmms_eval_out/plan.json`으로 체크포인트 plan에서 그대로 복사합니다.

| 이름 | 방법 | 논문 / 공식 코드 |
|---|---|---|
| `selfval_bon` | Self-Validation, Best-of-N (N=10) | arXiv 2601.22451, github.com/Liushiyu-0709/SelfVal (`llava/eval/selection.py`, `myutils.py`) |
| `selfval_fta` | Self-Validation, Filter-then-Aggregate (N=3) | 같은 논문, `llava/eval/aggreagate.py` |
| `logiccheck` | LogicCheckGPT (logical closed loop) | arXiv 2402.11622 (ACL 2024), github.com/CRIPAC-DIG/LogicCheckGPT (`logiccheckgpt/check_llava.py`) |

공정성 원칙: 두 방법 모두 **공식 코드의 기본값**을 쓰고 튜닝하지 않았습니다(우리 쪽으로도, 불리한 쪽으로도). 프롬프트는 공식 코드에서 글자 그대로 복사했습니다. 어쩔 수 없이 바꾼 부분은 아래 "공식 코드와 다른 점"에 전부 적었고, 같은 목록이 각 칸의 `run.log`에 찍힙니다. 각 칸 폴더의 `prompting_trace.jsonl`에 항목별 중간 단계(후보 캡션, 객체, LPFV 확률, 선택 결과 / 추출 엔티티, 질문, 답, closed-loop 비율, 보충 정보, 최종 답)가 남아서 리뷰어에게 그대로 보여줄 수 있습니다.

## 어느 벤치마크에 적용되나

| 벤치마크 | selfval_bon / selfval_fta | logiccheck |
|---|---|---|
| COCO CHAIR (coco_cap_chair) | 적용 | 적용 |
| DOCCI 캡셔닝 (docci_caption) | 적용 | 적용 |
| MM-Vet | 해당 없음 (표에서 빈칸) | 적용 (자유 답변을 고쳐 씀) |
| MME | 해당 없음 | 적용 (yes/no) |
| MMStar, MMBench | 해당 없음 | 실행은 하지만 사실상 base와 같음 (아래) |
| POPE, MMVP | 해당 없음 | 기본 실행에서 제외 (`BASELINE_SKIP_TASKS`; 비우면 함께 돔) |

- **Self-Validation은 캡셔닝 방법**입니다. 논문은 COCO 캡션(CHAIR, F1, GPT 점수)만 평가하고, 방법 자체가 "캡션 후보 N개 → 객체 검증 → 선택/집계"라서 객관식·yes/no에는 정의되지 않습니다. 그래서 캡션 그룹에서만 돌리고, 다른 그룹 칸은 만들지 않습니다(`report_all` 표에서 "Not applicable"로 표시).
- **LogicCheckGPT는 답변을 사후 수정하는 방법**이라 어떤 자유 답변에도 적용됩니다(논문: POPE, MME 일부, 자유 서술). MMStar·MMBench는 lmms-eval 프롬프트가 보기 글자만 답하게 해서("B") 답에서 객체가 나오지 않고, 공식 코드 흐름대로면 답이 그대로 나갑니다. 실제 숫자로 보여주려고 돌리긴 하지만 base와 거의 같게 나오는 게 정상입니다.

## Self-Validation: 구현 내용 (공식 코드 기준)

1. **후보 샘플링**: 태스크 프롬프트 그대로, multinomial sampling으로 N개(BoN 10, FtA 3; 논문 Table 1 설정), temperature 0.5, top-k 50.
2. **객체 추출**: 기본값 `--extract-method mscoco` = CHAIR의 `caption_to_words`(MSCOCO 80 클래스 + 동의어). lmms-eval `coco_cap_chair`의 같은 함수를 불러 씁니다(공식 코드도 CHAIR evaluator pickle의 같은 함수). DOCCI에도 같은 목록을 씁니다(논문 기본값).
3. **LPFV 검증**: 같은 이미지에 새 단일 턴 프롬프트 `"Describe any element of the image with only one word or phrase."`를 주고, 객체 이름(`obj.capitalize()`)의 토큰을 teacher forcing했을 때 각 토큰 확률의 곱. 확률은 공식 코드처럼 sampling warper(top-k 50, temperature 1)를 거친 softmax입니다(top-50 밖 토큰은 0).
4. **BoN**: 캡션 점수 = 객체 확률 평균(객체가 없으면 1.0, 공식 코드), argmax 캡션 선택.
5. **FtA**: 확률 ≤ α인 객체 단어를 `[IDX]`로 바꾸고 구두점(쉼표 포함) 뒤에서 잘라 `[IDX]`가 든 조각을 버림 → 남은 조각을 `||`로 이어 공식 집계 프롬프트(`Description from different sources: ... Given the materials above, give the final description:`)로 이미지와 함께 greedy 생성.

## LogicCheckGPT: 구현 내용 (check_llava.py 기준)

1. base 모델의 greedy 답(= base 칸의 답과 같은 lmms-eval 경로).
2. 첫 문장에 No/no/not이 있으면 그대로 종료(`detect_non_exist`).
3. LLM으로 대명사 치환 → 문장 분리(spaCy) → 문장별 엔티티 추출.
4. 엔티티마다 `"Could you please describe the {entity} in the picture? ..."`를 LVLM에 greedy 1번 + 샘플 2번(temperature 1).
5. 답에서 속성 claim 추출(`original&modified`), identity 노출·무관 질문 제거, 5개 초과가 될 때까지 → `"Could you tell me all the objects that ... in the image?"`로 바꿔 LVLM이 greedy로 답.
6. closed-loop 확인: 답에 엔티티 단어가 있으면 Yes, 아니면 LLM에 Yes/No(temperature 0).
7. Yes 비율 ≥ 0.4(코드 기본값)면 존재. `There is (no) X.` 보충 정보로 LLM이 원래 답을 고침 → 최종 답.

텍스트 LLM 설정은 코드 그대로: 시스템 프롬프트 6종, temperature 0.2(확인 단계 0.0), max_tokens 1024. LVLM 답은 최대 512 토큰.

## 공식 코드와 다른 점 (전부)

공통
- LVLM이 LLaVA-1.5 대신 **Qwen2.5-VL-7B-Instruct**(우리 base)이고 Qwen chat template을 씁니다. 두 논문 모두 Qwen 계열도 평가했습니다(Self-Validation: Qwen2.5-VL-7B, LogicCheckGPT: Qwen-VL-Chat).
- 이미지는 lmms-eval base 경로와 똑같이(JPEG data URL, 같은 min/max pixels) 넣습니다.
- 생성 batch 처리(여러 항목·질문을 한 번에). 각 항목의 계산은 항목별 코드와 같습니다. 샘플링 seed는 plan seed + 문서 id로 고정.

Self-Validation
- 후보와 최종 캡션 길이: 우리 평가의 캡션 한도(plan의 `max_new_tokens`, 학습 한도 128)를 모든 모델과 똑같이 씁니다. 공식 코드는 512. 베이스라인만 더 길게 쓰게 하면 CHAIR/Recall 비교가 길이 차이로 흔들리기 때문입니다. 필요하면 `--caption_max_new_tokens`를 올린 plan으로 체크포인트와 베이스라인을 함께 다시 돌리면 됩니다.
- 샘플링: temperature 0.5 + top-k 50 + top-p 1을 명시합니다(공식 코드가 LLaVA에서 쓰는 HF 기본값). 샘플링 설정을 비워 두면 HF가 모델 폴더의 `generation_config.json` 값을 상속하는데, Qwen 계열은 이 값이 거의 greedy에 가깝게(top-k/top-p가 매우 작게) 잡혀 있어 N개 후보가 거의 같아질 수 있습니다. 그래서 공식 코드가 실제로 쓴 값을 명시했습니다. repetition_penalty는 base 평가와 같게 모델 기본값(1.05)을 그대로 둡니다.
- LPFV를 객체마다 generate를 따로 부르는 대신, 이미지 하나당 고유 객체 전부를 한 번의 forward로 계산(같은 확률; `prompting_smoke.py`가 단계별 계산과 같은지 확인).
- **2026-11-10 버그 수정:** 이 LPFV forward가 processor의 `mm_token_type_ids`를 모델에 넘기지 않았습니다. transformers 5.x는 그 값이 없으면 이미지 토큰의 3-D(mRoPE) 위치를 못 만들고 1-D 위치를 써서, 객체 확률이 이미지 위치 정보가 틀린 채로 계산됐습니다. 후보 캡션 생성(`generate`)은 처음부터 정상이었습니다. 수정 후 `prompting_smoke.py`는 이 forward의 확률이 `generate()` 첫 단계 확률과 같은지도 확인합니다(옛 코드는 이 검사에서 걸림). **2026-11-08 리포트의 `selfval_bon` / `selfval_fta` 숫자는 수정 전 코드로 나온 것이라 다시 돌려야 합니다.** LogicCheckGPT는 `generate`만 써서 영향이 없습니다.
- FtA의 α: 논문 본문에 주 표 값이 없어 ablation에서 고정한 0.01을 씁니다.
- 논문의 GPT-4o-mini 정확도/관련도 점수는 재현하지 않습니다(우리 표의 지표로 비교).

LogicCheckGPT
- 텍스트 LLM: gpt-3.5-turbo-0125 대신 **로컬 judge 서버(Qwen2.5-14B-Instruct, vLLM)**. 프롬프트·온도·토큰 한도는 같습니다. 논문 부록(Table 4)에서 Vicuna-13B로 바꿔도 GPT-3.5와 비슷했다고 보고합니다. OpenAI 키가 있으면 `--prompting_args '{"logiccheck": {"llm_url": "https://api.openai.com/v1", "llm_model": "gpt-3.5-turbo-0125"}}'`(+ `OPENAI_API_KEY`)로 원래 설정 그대로 돌릴 수 있습니다.
- 문장 분리: spaCy `en_core_web_lg`(코드와 같음), 없으면 `en_core_web_sm`, 없으면 정규식. 서버에는 `python -m spacy download en_core_web_lg`를 권장합니다.
- 추출기가 "아무것도 없음"으로 답한 `None`은 엔티티에서 빼고, 엔티티가 하나도 없으면 답을 그대로 둡니다. 공식 코드는 `None`이라는 객체에 대해 질문을 만들어 버리는데, 이는 의도된 동작이 아니라고 판단했습니다.
- **yes/no 답 보정**: Qwen은 POPE/MME 프롬프트("Please answer yes or no.")에 "Yes" 한 단어로 답하는데, 논문의 LLaVA는 "Yes, there is a X in the image."처럼 문장으로 답했습니다. 공식 코드를 그대로 쓰면 "Yes"에서 객체가 안 나와 방법이 아무것도 하지 않으므로, **답이 맨 "Yes"일 때만 질문 문장에서 엔티티를 추출**해 같은 closed loop를 돌립니다. 베이스라인이 이 태스크에서 제대로 작동하게 하려는 보정입니다(불리하게 하는 쪽이 아님).
- 임계값 0.4: 공식 코드 기본값. 논문은 모델별로 [0, 0.9]에서 고른 값을 보고(LLaVA-1.5 0.4, Qwen-VL-Chat 0.2)하는데, 평가 셋에서 고르는 것이라 하지 않았습니다. Qwen-VL-Chat 값으로 보려면 `--prompting_args '{"logiccheck": {"threshold": 0.2}}'`.
- LLM 호출 실패 시 공식 코드는 무한 재시도, 여기선 6번(최대 몇 분) 후 칸을 실패로 처리합니다.

## 실행 (서버, conda activate argmax, 코드 폴더에서)

```
pip install qwen-vl-utils openai nltk spacy && python -m spacy download en_core_web_lg
python prompting_smoke.py --out_dir $TMPDIR/prompting_smoke   # CPU, 몇 분: "[smoke] OK"로 끝나야 함
bash rl_paper_commands.sh judge             # LogicCheckGPT의 텍스트 LLM (MM-Vet judge와 같은 서버)
bash rl_paper_commands.sh baselines_quick   # 태스크당 4개로 모든 칸 점검 (수십 분)
bash rl_paper_commands.sh baselines 3       # 실제 실행, 워커 3개
bash rl_paper_commands.sh report_all        # lmms_eval_out/summary.md 에 베이스라인 열 추가
bash rl_paper_commands.sh report_prompting  # self 우위 벤치마크 3분류 비교: lmms_eval_baselines/prompting_comparison.md / .html
```

결과는 `lmms_eval_baselines/<이름>/<그룹>/`에 쌓이고(응답 캐시 포함, 중단돼도 이어서 실행), plan은 체크포인트 plan의 설정을 복사해 따로 고정됩니다.

## 비용 (대략, base 생성 대비)

- `selfval_bon`: 캡션 10개 + LPFV forward 1번 → 캡션 그룹 base의 약 10배.
- `selfval_fta`: 캡션 3개 + LPFV + 집계 1번 → 약 4배.
- `logiccheck`: 캡션 하나에 엔티티가 8개쯤이면 LVLM 답 약 70개(최대 512 토큰) + LLM 호출 30~40개 → 캡션 그룹에서 base의 수십 배로 가장 무겁습니다. MME·MMBench·MMStar는 대부분 조기 종료되거나 엔티티 1~2개라 가볍습니다. 시간이 부족하면 캡션 그룹만 먼저 돌리거나 `PROMPTING_GEN_BATCH`(기본 16)를 키우세요.
