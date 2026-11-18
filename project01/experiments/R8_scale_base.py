# R8: 규모 확인 — 제출 기준 SVC(오답 100 조합)를 학습 10만으로 (W09 학습 곡선, 13회차)
# 배경: 메인 세션이 24만 10-fold 실제 결과를 기록 → 기준 SVC 0.9964125, Stacking 0.9964625 (차이 +0.00005).
#       5만 기준에서는 Stacking 이 오답 100→80 이었는데 전체 데이터에서는 효과가 거의 사라짐.
#       그러면 15차 특성 SVC(5만 88, 10만 86 — R5)의 개선도 규모가 커지면 사라지는지 확인해야 한다.
# 방법: R5 와 같은 조건(학습 앞 10만, 채점 220000~240000, C=10)에서 기준 조합(보정 픽셀 + HOG42 + HOG84, gamma 1.0)
#       오답을 재서 R5 10만 결과(86)와 비교. 격차가 10개 이상이면 15차 특성은 전체 데이터에서도 유효 → 제출 후보.
# 규칙: 원본 24만 안의 앞부분을 더 쓰는 것뿐(증강 아님), PCA·StratifiedKFold 없음, 학습·채점 구간 겹치지 않음.
# 사용: python3 R8_scale_base.py 100000   /   python3 R8_scale_base.py smoke
import sys, os, time, resource
import numpy as np
from collections import Counter
import common
NMAX = 100000
common.TR = slice(0, NMAX)
common.CACHE = os.path.join(common.ROOT, "cache_r8"); os.makedirs(common.CACHE, exist_ok=True)
SPEC = [("dkpix", 1), ("h42", 1), ("h84", 1)]
OUT = os.path.join(common.ROOT, "..", "results", "R8_scale_base.md")

def main():
    arg = sys.argv[1]
    from sklearn.svm import SVC
    d = common.load_raw()
    y_all = np.asarray(d[0:NMAX, 0]); yte = np.asarray(d[220000:240000, 0])
    t0 = time.time()
    A_all = common.block("tr", SPEC); B = common.block("te", SPEC)
    print(f"features {A_all.shape} {B.shape} ready in {time.time()-t0:.0f}s", flush=True)
    sizes = [2000] if arg == "smoke" else [int(s) for s in arg.split(",")]
    for n in sizes:
        A = A_all[:n]; y = y_all[:n]
        Bs, ys = (B[:500], yte[:500]) if arg == "smoke" else (B, yte)
        t = time.time()
        m = SVC(C=10, gamma=1.0 / (A.shape[1] * A.var()), cache_size=1000).fit(A, y)
        fit_s = time.time() - t
        pr = m.predict(Bs); wr = int((pr != ys).sum()); tot = time.time() - t
        pairs = Counter((int(a), int(b)) for a, b in zip(ys[pr != ys], pr[pr != ys])).most_common(4)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        line = (f"| 기준 SVC | {n:,} | {wr} | {1 - wr / len(ys):.4f} | {fit_s:.0f}s | {tot:.0f}s | "
                f"{len(m.support_):,} | {mem}MB | {pairs} |")
        print("RESULT", line, flush=True)
        if arg != "smoke":
            new = not os.path.exists(OUT)
            with open(OUT, "a") as f:
                if new:
                    f.write("| 모델 | 학습 개수 | 오답 (2만 중) | 정확도 | 학습 시간 | 학습+예측 | 서포트 벡터 수 | 최대 메모리 | 오답쌍 상위 (정답,예측) |\n"
                            "|---|---|---|---|---|---|---|---|---|\n")
                f.write(line + "\n")

if __name__ == "__main__":
    main()
