# R9: 큰 학습(10만)에서 gamma 키우기 (W11 RBF gamma: 데이터가 많을수록 더 좁은(큰 gamma) 커널이 버틸 수 있음, 14회차)
# 배경: 5만 기준 gamma 탐색(실험 B·E)은 1.0~1.2 가 최선이었지만, 데이터가 늘면 최적 gamma 가 커지는 경향이 있다.
#       10만 결과: 기준 SVC g1.0 83(R8), 15차 특성 SVC g1.2 86(R5). 메인 세션 12만: 기준 86, svc15 79.
#       → 2만 채점에서 ±7 정도는 우연 범위. gamma 를 크게 바꿔야(2배) 차이가 보일 것.
# 방법: 학습 앞 10만, 채점 220000~240000, C=10. (svc15 g2.0), (기준 g2.0), (svc15 g1.6) 을 차례로 1개씩.
# 규칙: 원본 24만 안의 앞부분 사용(증강 아님), PCA·StratifiedKFold 없음, 레이블은 학습에만, 학습·채점 구간 겹치지 않음.
# 사용: python3 R9_gamma_big.py all   /   python3 R9_gamma_big.py smoke
import sys, os, time, resource, gc
import numpy as np
from collections import Counter
import common
NMAX = 100000
common.TR = slice(0, NMAX)
common.CACHE = os.path.join(common.ROOT, "cache_r9"); os.makedirs(common.CACHE, exist_ok=True)
import R4_skeleton as r4                         # base() = 15차 prof 조합 (svc15)
OUT = os.path.join(common.ROOT, "..", "results", "R9_gamma_big.md")
CONFIGS = [("svc15", 2.0), ("기준", 2.0), ("svc15", 1.6)]

def feats(kind, part):
    if kind == "svc15": return r4.base(part).astype(np.float64)
    return common.block(part, [("dkpix", 1), ("h42", 1), ("h84", 1)])

def main():
    smoke = sys.argv[1] == "smoke"
    from sklearn.svm import SVC
    d = common.load_raw()
    y_all = np.asarray(d[0:NMAX, 0]); yte = np.asarray(d[220000:240000, 0])
    for kind, gm in CONFIGS:
        t0 = time.time()
        A = feats(kind, "tr"); B = feats(kind, "te"); y = y_all
        if smoke: A, y, B, ys = A[:2000], y_all[:2000], B[:500], yte[:500]
        else: ys = yte
        t = time.time()
        m = SVC(C=10, gamma=gm / (A.shape[1] * A.var()), cache_size=1000).fit(A, y)
        fit_s = time.time() - t
        pr = m.predict(B); wr = int((pr != ys).sum()); tot = time.time() - t0
        pairs = Counter((int(a), int(b)) for a, b in zip(ys[pr != ys], pr[pr != ys])).most_common(4)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        line = (f"| {kind} | {gm} | {len(y):,} | {wr} | {1 - wr / len(ys):.4f} | {fit_s:.0f}s | {tot:.0f}s | "
                f"{len(m.support_):,} | {mem}MB | {pairs} |")
        print("RESULT", line, flush=True)
        if not smoke:
            new = not os.path.exists(OUT)
            with open(OUT, "a") as f:
                if new:
                    f.write("| 특성 | gamma 배수 | 학습 개수 | 오답 (2만 중) | 정확도 | 학습 시간 | 특성+학습+예측 | 서포트 벡터 수 | 최대 메모리 | 오답쌍 상위 (정답,예측) |\n"
                            "|---|---|---|---|---|---|---|---|---|---|\n")
                f.write(line + "\n")
        del A, B, m; gc.collect()

if __name__ == "__main__":
    main()
