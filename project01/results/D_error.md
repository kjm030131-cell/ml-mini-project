# 실험 D 오류 보고: StackingClassifier 실행 실패 (메모리 부족 + 실행 시간 제한)

- 날짜: 2026-11-17 (세션 D, 예약 실행)
- 브랜치: `exp/D-stacking` (exp-infra 기준)
- 결과: **D_1 (SVC+LGBM), D_0 (SVC+LGBM+원본SVC) 모두 결과 없음.** D_1이 3번 연속 메모리 부족(OOM)으로 강제 종료되어 D_0은 실행하지 않음.

## 무슨 일이 있었나

| 시도 | 설정 | 경과 | 결과 |
|---|---|---|---|
| D_1 시도 1 | 원본 (LGBM n_jobs=2) | 1590초 | OOM Killed (anon-rss 6.10GB) |
| D_1 시도 2 | LGBM n_jobs=1 | 6초 | OOM Killed (anon-rss 6.10GB) |
| D_1 진단 | LGBM n_jobs=1, 90초 제한 | 약 6~10초 | OOM Killed (anon-rss 6.10GB) |

커널 로그 (dmesg):
```
Memory cgroup out of memory: Killed process 336 (python3) total-vm:6618980kB, anon-rss:6101492kB
Memory cgroup out of memory: Killed process 581 (python3) total-vm:6572320kB, anon-rss:6101340kB
Memory cgroup out of memory: Killed process 730 (python3) total-vm:6572352kB, anon-rss:6100084kB
```
전체 로그: `D_1.out`

## 원인

1. **메모리: 이 환경의 사용 한도는 약 6.1GB** (컨테이너 전체 8GB, 셸 cgroup 기준).
   - `sD_stacking.py`는 모드와 관계없이 특성 6종을 모두 float64로 붙임: 학습 A = 50000×4936 (1.97GB), 채점 B = 20000×4936 (0.79GB). 적재 단계 최대 RSS 3.39GB (따로 측정함).
   - 학습을 시작하면 ColumnTransformer가 열을 잘라 복사하고(SVC 부분 50000×2629 float64 ≈ 1.05GB), SVC/libsvm이 한 번 더 복사함 → 6.1GB 넘음.
   - LGBM n_jobs를 1로 줄여도 소용없었음: 학습 시작 직후 SVC 단계에서 바로 죽음. 즉 LGBM 스레드 수 문제가 아님.
   - 시도 1만 26분까지 버틴 건 첫 실행 때 특성 캐시를 만들면서 메모리 사용 시점이 달랐기 때문으로 보임 (캐시가 생긴 뒤에는 몇 초 만에 죽음).
2. **실행 시간: 이 예약 세션에서는 30분 넘는 작업을 끝까지 돌릴 수 없음.**
   - Bash의 `run_in_background` 옵션이 없음 (`InputValidationError: An unexpected parameter run_in_background`).
   - 앞에서 실행하는 Bash는 최대 10분. 백그라운드 감시 도구(Monitor)는 최대 30분이 지나면 프로세스를 종료함.
   - 프로세스를 떼어 내 실행(setsid/nohup류)하는 방식은 지시 위반이고 권한도 거부됨.
   - D_1, D_0은 각각 1시간 안팎 걸리므로 메모리 문제를 고쳐도 이 환경에서는 끝까지 돌릴 수 없음.

## 제안 (결과가 바뀌지 않는 메모리 절감)

`sD_stacking.py`만 고치면 됨 (세션 D 파일이므로 규칙상 수정 가능):
- 모드 1에서는 실제로 쓰는 `dkpix, h42, h84`만 붙이기 (2629열 → A 1.05GB). ColumnTransformer가 어차피 버리는 열이므로 결과는 같음.
- 채점 행렬 B는 `clf.fit` 다음에 만들기 (학습 중 0.79GB 절약).
- 실행은 메모리 10GB 이상, 1시간 넘게 돌릴 수 있는 곳(로컬 PC나 Colab CPU 런타임)에서 하는 것을 권장. 두 모드를 동시에 돌리면 안 됨.

참고: 시도 2 때 지시대로 `sD_stacking.py`의 LGBM `n_jobs=2 → 1`로 고쳤지만, 효과가 없어서 원래대로 되돌림. 이 PR에는 results의 D_ 파일만 들어 있음.

## 비교 기준 (참고)
| 실험 | 오답 (2만 중) | 정확도 | 시간 |
|---|---|---|---|
| 기준선 SVC 단독 | 100 | 0.9950 | - |
| 이전 수동 Stacking SVC+LGBM | 89 | 0.99555 | - |
| 이전 수동 Stacking SVC+LGBM+원본SVC | 85 | 0.99575 | - |
| StackingClassifier(SVC+LGBM, cv=3) | 실패 (OOM) | - | - |
| StackingClassifier(SVC+LGBM+원본SVC, cv=3) | 미실행 | - | - |
