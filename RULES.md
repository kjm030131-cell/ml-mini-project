# 미니 프로젝트 규칙 (2026-11-26 교수님 공지·중간 리뷰 반영, 최우선)

이 파일이 다른 모든 문서(README, EXPERIMENTS, RESEARCH, HUB)보다 우선한다. 충돌하면 이 파일을 따른다.

## 1. 교차검증은 반드시 shuffle=True (전 프로젝트)
- 교수님 공지: 데이터 순서에 따라 fold 분포가 달라지므로 `shuffle=True` 로 교차검증하고 그 결과를 제출할 것
- 셔플 없이 점수가 더 높게 나와도 더 좋은 모델로 보지 않음
- 코드: `kf = KFold(n_splits=10, shuffle=True, random_state=42)` 를 만들고 `cross_val_score`, `cross_val_predict`, `GridSearchCV`, `RandomizedSearchCV`, `StackingClassifier/Regressor(cv=...)`, `learning_curve` 의 `cv=` 에 **모두** KFold 객체를 넘긴다
- **`cv=10`, `cv=5` 같은 정수 금지** (분류에서는 StratifiedKFold, 셔플 없음)

## 2. 중간 리뷰(Mini Project Review) 지적 사항
| 프로젝트 | 금지 |
|---|---|
| 1번 EMNIST | **HOG(Histogram of Oriented Gradients) 사용 제한** — `hog`, `hog_block` 등 기울기 방향 히스토그램 특성 금지 |
| 2번 HAR | **데이터 누수**: 전체 데이터(또는 다른 행)에서 뽑은 통계 특성을 검증에 쓰지 않는다. 다른 프로젝트 결과물 제출 금지 |
| 3번 집값 | **행(인스턴스) 임의 제거 금지** (21,613행 전부 사용). 10-fold 결과·평균 출력 누락 금지. 제출 파일명 형식 준수 |

## 3. 기존 공통 규칙 (W15 5쪽)
- GPU 금지, XGB·LGBM 은 CPU 버전
- 허용 모델: SGD, LogisticRegression, LinearSVC/SVR, SVC/SVR, DecisionTree, Voting, Bagging, RandomForest, ExtraTrees, AdaBoost, GradientBoosting, Stacking, XGB, LGBM
- 사용 가능: Grid Search, Random Search, Stacking / 사용 불가: Resample(증강), PCA, StratifiedKFold
- 제출 파일명: `mini_proj_0X_20221329.ipynb`, 모든 셀 실행 후 10-fold CV 결과와 평균 출력

## 4. 누수 방지 원칙 (이 저장소에서 지키는 방식)
- 특성은 **행 하나(이미지 한 장, 측정 구간 하나, 집 한 채)만 보고** 계산한다. 다른 행·전체 평균·정답을 쓰지 않는다
- 스케일링 등 데이터로 fit 하는 전처리는 `Pipeline` 안에 넣어 fold 마다 학습 데이터로만 fit
- 2번: 같은 사람의 앞뒤 구간 이동평균(rolling) 특성은 시험 fold 의 다른 행을 쓰므로 **사용하지 않음**
- gamma 등 하이퍼파라미터를 전체 데이터 분산으로 계산하지 않고 Grid Search 로 고른다

## 5. 현재 제출 노트북 상태 (2026-11-28 갱신)
| 프로젝트 | 10-fold 평균 (shuffle=True) | 모델 | 상태 |
|---|---|---|---|
| 1 | 미실행 (예상 약 0.994) | deskew 픽셀 + 획 특성 + 윤곽 프로파일 → SVC, HOG 없음 | 맥에서 실행 필요 (약 수 시간) |
| 2 | **0.9958** (정확도) | 원본 561 특성 → Stacking(SVC C=30 g=0.02 + LGBM 1500) → LogisticRegression | 실행 완료 (이전 SVC 단독 0.9931) |
| 3 | **106,003** (RMSE) | XGB·LGBM(원래/log 가격) 4개 Stacking, 21,613행 전부 | 실행 완료 (추가 연구에서 개선 없음) |
| 4 | **0.5429** (RMSE) | Stacking(ExtraTrees + XGB + RF + SVR) → LinearRegression(positive) | 실행 완료 (이전 Voting 0.5446) |

## 6. 예전 실험 기록에 대해
- `project01/experiments`, `project01/results`, `exp/*` 브랜치, `RESEARCH.md` 의 1번 실험은 **대부분 HOG 를 쓰고 셔플 없는 비교**라 위 규칙에 맞지 않는다. 참고용 기록일 뿐 제출 근거로 쓰지 않는다
- HOG 를 쓴 예전 1번 노트북은 `project01/archive_hog/` 로 옮겼다 (**제출 금지**)
- 2·3·4번 `experiments` 폴더의 스크립트도 정수 cv(셔플 없음) 기준이라 참고용
