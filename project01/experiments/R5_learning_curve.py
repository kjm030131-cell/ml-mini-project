# R5: 학습 곡선 (W09 35~36쪽 learning_curve 개념, W11 16쪽 큰 데이터 SVC 학습 시간)
# 목적: 지금까지 모든 비교는 '학습 5만'이지만 실제 제출은 24만 10-fold(학습 21.6만). 15차 단일 SVC(오답 88)를
#       학습 1.25만 / 2.5만 / 5만 / 10만 개로 늘려 가며 같은 채점 2만(220000~240000)에서 오답과 학습 시간을 잰다.
#       → (1) 21.6만 학습 시 예상 정확도(외삽), (2) 단일 SVC 전체 10-fold 예상 시간,
#         (3) Stacking(5만 기준 78~81)과 단일 SVC 의 격차가 데이터가 늘면 줄어드는지 판단 근거.
# 규칙: 데이터 개수를 늘리는 것이 아니라 원본 앞부분을 더 많이 쓰는 것(원래 24만 안), 증강·PCA·StratifiedKFold 없음.
#       학습 구간(앞 10만)과 채점 구간(220000~240000)은 겹치지 않음.
# 사용: python3 R5_learning_curve.py 100000      (한 크기)
#       python3 R5_learning_curve.py 12500,25000,50000
#       python3 R5_learning_curve.py smoke
import sys, os, time, resource
import numpy as np
from collections import Counter
import common
NMAX = 100000
common.TR = slice(0, NMAX)                       # feat() 가 앞 10만 개 특성을 만들도록
common.CACHE = os.path.join(common.ROOT, "cache_r5"); os.makedirs(common.CACHE, exist_ok=True)
import R4_skeleton as r4                         # base() = 15차 prof 조합 (feat 를 통해 위 설정을 따름)

OUT = os.path.join(common.ROOT, "..", "results", "R5_learning_curve.md")

def main():
    arg = sys.argv[1]
    from sklearn.svm import SVC
    d = common.load_raw()
    y_all = np.asarray(d[0:NMAX, 0]); yte = np.asarray(d[220000:240000, 0])
    t0 = time.time()
    A_all = r4.base("tr"); B = r4.base("te").astype(np.float64)
    print(f"features {A_all.shape} {B.shape} ready in {time.time()-t0:.0f}s", flush=True)
    sizes = [2000] if arg == "smoke" else [int(s) for s in arg.split(",")]
    for n in sizes:
        A = A_all[:n].astype(np.float64); y = y_all[:n]
        Bs, ys = (B[:500], yte[:500]) if arg == "smoke" else (B, yte)
        t = time.time()
        m = SVC(C=10, gamma=1.2 / (A.shape[1] * A.var()), cache_size=1000).fit(A, y)
        fit_s = time.time() - t
        pr = m.predict(Bs); wr = int((pr != ys).sum()); tot = time.time() - t
        pairs = Counter((int(a), int(b)) for a, b in zip(ys[pr != ys], pr[pr != ys])).most_common(4)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        line = (f"| {n:,} | {wr} | {1 - wr / len(ys):.4f} | {fit_s:.0f}s | {tot:.0f}s | "
                f"{len(m.support_):,} | {mem}MB | {pairs} |")
        print("RESULT", line, flush=True)
        if arg != "smoke":
            new = not os.path.exists(OUT)
            with open(OUT, "a") as f:
                if new:
                    f.write("| 학습 개수 | 오답 (2만 중) | 정확도 | 학습 시간 | 학습+예측 | 서포트 벡터 수 | 최대 메모리 | 오답쌍 상위 (정답,예측) |\n"
                            "|---|---|---|---|---|---|---|---|\n")
                f.write(line + "\n")
        del A

if __name__ == "__main__":
    main()
