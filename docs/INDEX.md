# 자료 요약 (Claude가 정리한 노트)

강의자료 원본은 저장소에 올리지 않는다. 아래는 실험에 필요한 내용만 정리한 요약이다.

## 미니 프로젝트 문제
| # | 주제 | 지표 | 데이터 |
|---|---|---|---|
| 1 | EMNIST 손글씨 숫자 | 10-fold CV 정확도 ↑ | `data/emnist_digits_train_part*.npz` (24만×785) |
| 2 | 스마트폰 행위 인식(HAR) | 10-fold CV 정확도 ↑ | `data/project02/train.csv.gz` (7,352×563, subject·Activity 제외 561 특성, `pd.read_csv`로 바로 읽힘) |
| 3 | King County 집값 | 10-fold CV RMSE ↓ | `data/project03/kc_house_data.csv` (21,613×21, price·date 제외 19 특성) |
| 4 | 레드와인 품질 | 10-fold CV RMSE ↓ | `data/project04/winequality-red.csv` (1,599×12) |

- 공통 규칙: GPU 금지, 강의에서 다룬 알고리즘·모델·코드 안에서. 사용 가능 Grid/Random Search, Stacking. 사용 불가 Resample(증강), PCA, StratifiedKFold. XGB·LGBM은 CPU 버전
- 제출: `mini_proj_0X_20221329.ipynb`, 모든 셀 실행 후 10-fold 결과 배열과 평균 출력 필수. 지각 불가
- 랭킹 사이트: http://114.71.220.59:2025/ (학번 가입, 프로젝트 선택 후 점수와 ipynb 업로드)
- 슬라이드 기준선: 1번 SGD 0.8845, 2번 SGD 0.9193, 3번 LinearRegression RMSE 201,975, 4번 LinearRegression RMSE 0.6586

## 강의별 실험에 쓸 내용
| 강의 | 내용 |
|---|---|
| W01 | 특성 공학 = 특성 선택 + 특성 추출. 교수님은 학습에 도움 되는 특성을 찾는 작업 전반을 특성 추출로 봄 (45쪽) |
| W03 | 특성 조합으로 새 특성 만들기, 결측치 처리, 범주형 인코딩, MinMax·표준화 스케일링, Pipeline·ColumnTransformer, cross_val_score(cv=10)와 RMSE 계산 → 3·4번 회귀에 직접 사용 |
| W06 | cross_val_predict와 오차 행렬, 행 정규화 오차 행렬 시각화, OvR/OvO, SGD에 표준화 효과, 31쪽 구멍(동심원) 개수 특성 예시. 데이터 증식은 소개되지만 미니 프로젝트에서 금지 |
| W09 | 경사 하강법, 다항 회귀, 학습 곡선(learning_curve), 편향·분산, 로지스틱 회귀 → 분석 셀과 3·4번에 활용 |
| W11 | SVM은 특성 스케일에 민감해 StandardScaler 권장(5쪽), C·gamma 함께 조정(15쪽), 큰 데이터에서 SVC 학습 시간 급증(16쪽), SVR |
| W12 | 결정 트리 규제 하이퍼파라미터(max_depth, min_samples_leaf 등), 스케일링 불필요, 축 방향 민감성과 높은 분산 |

빠진 자료: 앙상블(RandomForest, Boosting, Stacking) 주차 강의
