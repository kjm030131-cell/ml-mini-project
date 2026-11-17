# 통합 명령 세션 안내 (HUB)

재민(사용자)이 편하게 말로 지시하면, 이 세션이 실험 현황 확인·정리·다음 실험 지시를 맡는다.
답은 짧은 한국어로. 표는 필요할 때만.

## 프로젝트
- 순천향대 머신러닝(윤재석 교수) 15주차 미니 프로젝트 1번: EMNIST 손글씨 숫자, 24만 개, **10-fold CV 평균 정확도**
- 제출 파일: `mini_proj_01_20221329.ipynb` (모든 셀 실행, 10-fold 결과 배열과 평균 출력 필수)
- 규칙(W15 5쪽): GPU 금지, 허용 모델만(SGD, LogisticRegression, LinearSVC, SVC, DecisionTree, Voting, Bagging, RandomForest, ExtraTrees, AdaBoost, GradientBoosting, Stacking, XGB·LGBM은 CPU), 증강·PCA·StratifiedKFold 금지
- 현재 랭킹 1등: 0.9816

## 현재 최고 조합 (제출용)
- SVC(C=10, RBF) + 기울기 보정 픽셀 784 + 촘촘한 HOG(4칸, 2이동) 1521 + 큰 칸 HOG(8칸, 4이동) 324
- 로컬 M4 맥 10-fold: fold 7개까지 0.996~0.997, 예상 평균 약 0.9964 (재실행 중)
- Kaggle 노트북: kimjaemin12/notebookdcc142d77c, Version 3 실행 중 (fold 2개 동시)

## 실험 비교 기준 (EXPERIMENTS.md)
- 학습 앞 5만 / 채점 220000~240000 (2만), 지표는 오답 수. 기준선 100개. 10개 이상 줄어야 개선
- 지금까지: 큰 칸 HOG 비중 2배 96개, SVC+LGBM Stacking 89개, SVC+LGBM+원본SVC Stacking 85개 (전체 실행 시 17~20시간 예상)

## 병렬 실험 세션 (2026-11-17 14:15 시작)
- A 특성 비중(`sA_weights.py`) → `project01/results/A_feature_weights.md`, 브랜치 exp/A-feature-weights
- B C·gamma(`sB_cgamma.py`) → `B_C_gamma.md`, exp/B-C-gamma
- C 추가 특성(`sC_features.py`) → `C_extra_features.md`, exp/C-extra-features
- D Stacking 재현(`sD_stacking.py`) → `D_stacking.md`, exp/D-stacking

## 자료
- `docs/INDEX.md`: 미니 프로젝트 4개 문제와 강의 요약 노트 (강의 원본은 저장소에 없음)
- 재민의 맥북 바탕화면 폴더 구조
  - `머신러닝강의자료`: 강의 PDF (W01, W03 3개, W06, W09, W11. W12는 아직 없음)
  - `머신러닝프로젝트`: 미니 프로젝트 문제 PDF 01~04, 랭킹 사이트 안내 PDF, emnist-digits-train.csv
  - `ML1`: 1번 제출 노트북 `mini_proj_01_20221329.ipynb`와 데이터 (터미널 nbconvert로 실행 중이면 끝날 때까지 건드리지 말 것)
  - `ML2`, `ML3`, `ML4`: 2~4번 프로젝트용 (재민이 만들 예정, 데이터 CSV가 들어갈 곳)
- 강의 원본은 재민의 맥북 `/Users/jaemin/Desktop/머신러닝강의자료` 에 있다. 이 세션이 재민의 컴퓨터와 연결돼 있으면(mcp__remote-devices__ 도구가 보이면) device_list_dir / device_stage_files 로 필요한 PDF만 가져와 읽는다. 연결돼 있지 않으면 재민에게 "데스크톱 앱에서 이 세션 열고 '이 컴퓨터에 연결' 후 + 메뉴의 폴더 추가로 머신러닝강의자료 폴더를 연결해 줘"라고 요청한다. 먼저 INDEX.md 요약으로 충분한지 보고, 원본은 꼭 필요할 때만 연다.
- 데이터: 1번은 `data/emnist_digits_train_part*.npz`, 2~4번은 재민에게 받아야 함

## 할 수 있는 일
- 현황 확인: `gh api repos/kjm030131-cell/ml-mini-project/pulls?state=all` 과 각 브랜치의 results 파일 읽기 (GraphQL은 막혀 있어 REST만)
- 정리: 기준선 100 대비 비교표, 효과 있는 조합 표시
- 새 실험: exp-infra 브랜치에 스크립트 추가(공통 함수 `common.py`의 `run_svc`) 후, create_trigger(run_once_at)로 새 실험 세션 생성. 프롬프트는 기존 A~D 형식 그대로
- 무거운 계산은 이 세션에서 직접 돌리지 말고 실험 세션에 맡긴다
