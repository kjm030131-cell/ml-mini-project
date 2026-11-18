# 3번 King County 집값 1차 연구 (2026-11-18)

평가: 21,613개, `cross_val_score(cv=10, scoring="neg_mean_squared_error")` → RMSE 평균 (낮을수록 좋음)

| 조합 | 10-fold RMSE |
|---|---|
| 슬라이드 LinearRegression (id 포함) | 201,975 (슬라이드) / 365,061 (sklearn 1.9, id 값이 커서 계산 불안정) |
| LinearRegression, id 제외 | 201,997 |
| RandomForest 300 | 126,265 |
| LGBM 기본 특성 | 115,145 |
| LGBM + 날짜·나이 | 112,642 |
| XGB + 날짜·나이 | 112,044 |
| XGB + 날짜·나이 + 위치(도심 거리, 회전 좌표, 면적 비율) | 107,860 |
| LGBM + 위치, log(price) 학습 | 107,264 |
| Voting XGB + LGBM(log) | 105,992 |
| **Stacking XGB·XGB(log)·LGBM·LGBM(log) → LinearRegression(positive)** | **105,563** |

- id 는 필요 없는 특성으로 제거 (슬라이드 5쪽 '필요 없어 보이는 특성은 제거')
- date 는 연·월·경과일로 숫자화 (슬라이드 7쪽 '관련성 높으면 숫자로 변환')
