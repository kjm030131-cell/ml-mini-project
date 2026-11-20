# 2번(HAR)·3번(집값) 누수 없는 연구 기록장

2026-11-20 시작. 1번 RESEARCH.md 와 같은 방식으로, 결과와 다음 할 일을 여기에 쌓는다.

## 누수 없는 평가 규칙 (2·3번 공통, 절대 규칙)
1. **바깥 10-fold 는 `KFold(n_splits=10)` 객체** (섞지 않음). 분류기에 `cv=10` 정수를 주면 사이킷런이 StratifiedKFold 를 써서 금지 규칙에 걸림. 회귀(3번)는 정수여도 KFold 지만 통일해서 객체로 쓴다
2. **정답(Activity, price)은 특성 계산에 쓰지 않는다**. 다른 행의 가격(같은 집 이전 거래가 등)도 특성으로 쓰지 않는다
3. **데이터 전체로 fit 하는 전처리 금지**: 스케일러, KMeans, 특성 선택은 모두 Pipeline 안에 넣어 fold 의 학습 행으로만 fit
4. **하이퍼파라미터 탐색도 fold 안에서 (중첩)**: 전체 데이터로 GridSearch/RandomizedSearch 를 한 뒤 그 설정으로 10-fold 를 돌리면, 채점 fold 를 보고 설정을 고른 셈이라 점수가 부풀려진다. 탐색기를 그대로 `cross_val_score` 에 넣거나(내부 cv=KFold(5)), 설정을 미리 고정한다
5. Stacking 내부 cv 도 `KFold(k)` 객체 (분류기에 정수 금지)
6. 행 섞기 금지: 2번은 50% 겹치는 창이라 섞으면 거의 같은 행이 학습·채점에 나뉘어 0.9996 까지 부풀려짐(1차 연구 확인)

## 기존 제출 노트북 누수 점검 (2026-11-20)
| 노트북 | 문제 | 고치는 법 |
|---|---|---|
| `mini_proj_02` 셀 '2) Grid Search' | 전체 7,352행으로 GridSearchCV → best_estimator_ 로 10-fold. **채점 fold 를 보고 C·gamma 를 고름(누수)** | 설정 고정 또는 `cross_val_score(GridSearchCV(..., cv=KFold(5)), ..., cv=KFold(10))` 중첩 |
| `mini_proj_02` 셀 '1)', '3)', cross_val_predict | `cv=10`·`cv=5` 정수 → **StratifiedKFold (금지)** | `KFold(n_splits=10)` 객체 |
| `mini_proj_03` 셀 '2) Randomized Search' | 전체 21,613행으로 RandomizedSearchCV → 그 설정으로 최종 10-fold. **같은 누수** | 중첩 탐색 또는 고정 설정 (중첩 점수 p3_r01 B) |
| `mini_proj_03` 특성 | 행 하나에서만 계산, 기준점·날짜는 상수 → 누수 없음 | 그대로 |
| `mini_proj_03` 데이터 | 같은 집(id)이 두 번 팔린 행 177쌍이 다른 fold 에 들어갈 수 있음. 정답 누수는 아니고 데이터 자체의 성질이라 행은 지우지 않음. 단 id·이전 거래가는 특성으로 쓰지 않음 | 그대로 |

## 2번 HAR: 두 등급
- **[S] 엄격**: 각 행의 561 특성만 사용. 행 순서·subject 를 전혀 쓰지 않음. 누수 논란 없음
- **[T] 문맥**: 같은 사람의 앞뒤 행 평균(rolling), 사람별 평균 빼기. 계산은 fold 에 들어온 행만으로(FunctionTransformer, 상태 없음), 정답 미사용 → 정답 누수는 아님. 단 subject·행 순서(시간 순서)를 쓰므로 **규칙 확인 필요** (교수님이 '각 행 독립 표본'을 의도했는지)
- 섞지 않은 KFold(10)은 fold 하나가 사람 2~4명 묶음 → 사실상 '처음 보는 사람' 평가. 그래서 이전 StratifiedKFold 점수(0.9576)보다 낮게 나옴

## 결과
### 2번 (7,352행, KFold(10) 섞지 않음, 기준 SGD 0.9193)
| 조합 | 10-fold 평균 | 최저 fold |
|---|---|---|
| [S] SVC C=10 | 0.9463 | 0.891 |
| **[S] SVC C=100** | **0.9482** | 0.894 |
| [S] StandardScaler + SVC C=10 | 0.9342 | 0.882 |
| [S] LogisticRegression C=1 | 0.9400 | 0.852 |
| [S] LinearSVC C=1 | 0.9438 | 0.856 |
| [S] 중첩 GridSearch SVC C(1~1000) | 0.9461 | 0.891 |
| [S] SVC C=100 gamma 0.3배 / 3배 | 0.9452 / 0.9480 | |
| [S] LGBM 500 | 0.9253 | 0.867 |
| [T] rolling 9+51 + SVC C=100 | 0.9695 | 0.917 |
| [T] 사람평균빼기 + rolling 9+51 | 0.9678 | 0.921 |
| **[T] 사람평균빼기 + rolling 5+75** | **0.9716** | 0.931 |

### 3번 (21,613행, KFold(10), RMSE)
| 조합 | 10-fold RMSE |
|---|---|
| LGBM log(price) 고정 설정 (기준) | 107,686 |
| + KMeans30 위치 유사도 (fold 안 fit) | 107,590 |
| **+ KMeans60 위치 유사도 (γ=500)** | **106,981** |
| + KMeans100 (γ=1000) | 107,394 |
| (이전 1차, 고정 설정이라 누수 없음) Stacking XGB·XGB(log)·LGBM·LGBM(log) → LR(positive) | 105,563 |

## 아이디어 대기열
### 2번
1. [S] 오차 행렬로 헷갈리는 쌍(앉기↔서기 예상) 확인 → 그 쌍을 가르는 행 단위 특성(중력 각도 angle(*,gravityMean) 비중 키우기 등)
2. [S] Stacking SVC+LR(+LGBM) 내부 KFold(5) — p2_r02 진행 중
3. [S] 특성 묶음(시간 영역 t*, 주파수 영역 f*, angle) 별 비중 조정 후 SVC
4. [T] 규칙 확인 뒤에만: rolling 창 조합, 문맥 + Stacking

### 3번
1. 중첩 RandomizedSearch 점수(정직한 튜닝 효과) — p3_r01 B 진행 중
2. KMeans60 유사도 특성을 1차 Stacking(105,563)에 넣기
3. zipcode 를 LGBM 범주형으로
4. 이상치 행(침실 33개 등)은 지우지 않음(채점 행이 바뀜), 대신 log 타깃 모델 비중

## 규칙 확인 필요 (재민에게)
- 2번 [T] 문맥 특성(같은 사람 앞뒤 행, 사람별 평균 빼기)을 써도 되는지. 정답은 안 쓰지만 subject·시간 순서를 씀
- 3번 KMeans 는 강의(W03 / 핸즈온 2장 ClusterSimilarity)에서 다룬 전처리로 보고 사용. 다른 의견이면 빼기

## 기록
### 2026-11-20 (재민 요청으로 시작)
- 기존 노트북 누수 점검: 2·3번 모두 '전체 데이터로 탐색 → 같은 데이터로 10-fold' 누수 발견, 2번은 정수 cv(StratifiedKFold)도 발견
- p2_r01·p2_r02, p3_r01 실행. 결과 위 표
