# self 우위 벤치마크: 체크포인트 vs 프롬프팅 베이스라인

> **스모크 테스트 결과입니다. 태스크당 4개만 돌렸으니 숫자를 해석하지 마세요.**

체크포인트 `lmms_eval_out`, 베이스라인 `lmms_eval_baselines_quick` (plan 2026-11-06 23:49:22).
굵은 글씨는 그 행에서 가장 좋은 값입니다. 화살표는 좋은 방향입니다.

> 미실행 칸: logiccheck/general, logiccheck/mcq

## self와 베이스라인 비교 요약

| 베이스라인 | self 이김 | 동점 | self 짐 |
|---|---|---|---|
| selfval_bon | 2 | 0 | 4 |
| selfval_fta | 0 | 0 | 6 |
| logiccheck | 3 | 0 | 7 |

## 1. base와 두 대조군을 모두 이김

| 벤치마크 | base | self | external | length | selfval_bon | selfval_fta | logiccheck | self 기준 |
|---|---|---|---|---|---|---|---|---|
| MMStar average ↑ | 0.6093 | **0.6127** | 0.6113 | 0.6107 | 해당 없음 | 해당 없음 | 0.0000 | logiccheck 이김 |
| MMStar fine-grained perception ↑ | 0.5160 | **0.5320** | 0.5160 | 0.5160 | 해당 없음 | 해당 없음 | 미실행 |  |
| MMBench ↑ | 83.33 | 83.51 | 83.16 | 83.25 | 해당 없음 | 해당 없음 | **100.00** | logiccheck 짐 |
| MME Perception ↑ | 1680.94 | **1690.00** | 1686.11 | 1676.26 | 해당 없음 | 해당 없음 | 미실행 |  |

## 2. base는 이기지만 대조군까지 다 이기지는 못함

| 벤치마크 | base | self | external | length | selfval_bon | selfval_fta | logiccheck | self 기준 |
|---|---|---|---|---|---|---|---|---|
| COCO Recall ↑ | 60.11 | 61.88 | 61.31 | 63.01 | **83.33** | **83.33** | **83.33** | selfval_bon 짐, selfval_fta 짐, logiccheck 짐 |
| MME Cognition ↑ | 618.57 | 621.07 | 623.21 | **625.36** | 해당 없음 | 해당 없음 | 50.00 | logiccheck 이김 |
| MMStar instance reasoning ↑ | 0.6840 | **0.6920** | **0.6920** | **0.6920** | 해당 없음 | 해당 없음 | 미실행 |  |
| MMStar logical reasoning ↑ | 0.6560 | **0.6600** | **0.6600** | 0.6560 | 해당 없음 | 해당 없음 | 미실행 |  |
| COCO CHAIR_i ↓ | 8.5106 | 8.4998 | 7.9523 | 7.6994 | 8.6957 | **0.0000** | 8.3333 | selfval_bon 이김, selfval_fta 짐, logiccheck 짐 |

## 3. base에는 지지만 대조군은 이김

| 벤치마크 | base | self | external | length | selfval_bon | selfval_fta | logiccheck | self 기준 |
|---|---|---|---|---|---|---|---|---|
| MM-Vet ↑ | 72.16 | 71.79 | 69.98 | 69.57 | 해당 없음 | 해당 없음 | **100.00** | logiccheck 짐 |
| DOCCI CAPTURE ↑ | **0.5955** | 0.5885 | 0.5930 | 0.5855 | 0.5686 | 0.5920 | 0.5553 | selfval_bon 이김, selfval_fta 짐, logiccheck 이김 |
| DOCCI BLEU-4 ↑ | 0.0477 | 0.0392 | 0.0466 | 0.0350 | 0.0677 | **0.0921** | 0.0578 | selfval_bon 짐, selfval_fta 짐, logiccheck 짐 |
| DOCCI METEOR ↑ | 0.1378 | 0.1238 | 0.1362 | 0.1190 | 0.1256 | **0.1577** | 0.1300 | selfval_bon 짐, selfval_fta 짐, logiccheck 짐 |
| DOCCI ROUGE-L ↑ | 0.2261 | 0.2192 | 0.2251 | 0.2152 | 0.2412 | **0.2695** | 0.2479 | selfval_bon 짐, selfval_fta 짐, logiccheck 짐 |

## 캡션 평균 단어 수

| 태스크 | base | self | external | length | selfval_bon | selfval_fta | logiccheck |
|---|---|---|---|---|---|---|---|
| COCO | 106.9 | 103.4 | 107.3 | 93.9 | 107.2 | 98.8 | 95.5 |
| DOCCI | 93.0 | 75.2 | 93.2 | 73.7 | 95.2 | 110.0 | 84.8 |

## 읽을 때 주의할 점

- 이 세 목록은 체크포인트 결과를 본 뒤 고른 것입니다. 논문에는 전체 표(`lmms_eval_out/summary.md`)와 함께 보고해야 합니다.
- 모든 값은 점 추정치입니다. 통계 검정을 하지 않았으므로 작은 차이는 우연일 수 있습니다.
- `selfval_bon`, `selfval_fta`는 캡션 방법이라 캡션 태스크(COCO, DOCCI)에만 적용됩니다.
- `logiccheck`는 MMStar, MMBench에서 답이 보기 글자뿐이라 사실상 base와 같게 나오는 것이 정상입니다.
- MM-Vet은 로컬 judge로 채점해서 절대값은 다른 논문과 비교할 수 없습니다.
- 공식 코드와 다른 점은 `BASELINES.md`에 있습니다. 캡션 길이 한도는 모든 모델이 128토큰입니다.
- 이 분류에 없는 pope, mmvp는 베이스라인을 돌리지 않았습니다.
