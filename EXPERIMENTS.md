# 병렬 실험 운영 규칙

> **[2026-11-26 규칙 변경] [RULES.md](RULES.md) 가 최우선**: 모든 교차검증 `KFold(shuffle=True)`, 1번 HOG 금지, 데이터 누수 금지, 행 제거 금지. 이 문서의 HOG·정수 cv 기준 내용은 예전 기록(참고용)이다.

여러 Claude 세션이 동시에 실험하고, 결과는 이 저장소에 모은다.

## 공통 기준
- 데이터: `data/emnist_digits_train_part*.npz` (교수님 지정 Kaggle crawford/emnist의 `emnist-digits-train.csv`와 동일, 24만 개)
- 비교 방법: 학습 = 앞 5만 개, 채점 = 220000~240000번 (2만 개). **지표는 2만 개 중 오답 수** (적을수록 좋음)
- 기준선: SVC(C=10) + 기울기 보정 픽셀 + 촘촘한 HOG(4,2) + 큰 칸 HOG(8,4) = **오답 100개**
- 개선 판정: 기준보다 **오답 10개 이상** 줄어야 진짜 개선으로 본다 (2~5개 차이는 우연 범위)
- 규칙(W15 슬라이드 5쪽): GPU 금지, 허용 모델 목록 안에서만, 증강·PCA·StratifiedKFold 금지

## 세션 작업 순서
1. `cd project01/experiments && pip install -q lightgbm` (필요 시)
2. 맡은 스크립트를 두 개 동시에 실행: `python sX.py 0 & python sX.py 1 & wait`
3. 결과는 `project01/results/<세션>.md`에 표 형식으로 자동 기록됨. 맨 위에 제목, 날짜, 한 줄 결론 추가
4. 브랜치 `exp/<세션>`으로 커밋하고 main 대상 PR 생성
5. 다른 세션 파일은 건드리지 않는다

## 결과 표 형식
| 실험 | 오답 (2만 중) | 정확도 | 시간 |
|---|---|---|---|
