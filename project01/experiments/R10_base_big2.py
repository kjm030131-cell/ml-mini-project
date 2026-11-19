# R10: 기준 SVC(제출 노트북 조합)를 메인 big2 조건(학습 0~120000, 채점 120000~220000, 10만 개)으로 (15회차)
# 배경: 메인 세션 big2(커밋 f934204)에서 같은 조건 svc15 오답 349, +HOG73 340, HOG12방향 380.
#       그런데 같은 10만 채점에서 '기준 SVC' 값이 없어 svc15 노트북이 기준 노트북보다 나은지 판단 불가.
#       2만 채점은 ±9 우연 범위라 10만 채점(오답 350 의 표준오차 약 ±19, 같은 문제를 푸는 짝 비교라 실제 잡음은 더 작음)으로 확정한다.
# 판정: 기준 오답이 349 보다 20개 이상 많으면 svc15 특성은 전체 데이터에서도 유효 → svc15 노트북 제출 권장.
#       ±20 안이면 차이 없음 → 더 단순·빠른 기준 노트북 유지.
# 규칙: 원본 24만 안에서 구간만 나눔(증강 아님), PCA·StratifiedKFold 없음, 레이블은 학습에만 사용, 학습·채점 구간 겹치지 않음.
# 사용: python3 R10_base_big2.py full   /   python3 R10_base_big2.py smoke
import sys, os, time, resource
import numpy as np
from collections import Counter
import common
from feats import deskew, hog2
OUT = os.path.join(common.ROOT, "..", "results", "R10_base_big2.md")

def feats(d, a, b):
    out = []
    for s in range(a, b, 10000):
        X = np.asarray(d[s:min(s + 10000, b), 1:]).astype(np.float32) / 255.0
        im = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
        out.append(np.hstack([im.reshape(len(im), -1), hog2(im, 4, 2), hog2(im, 8, 4)]).astype(np.float32))
    return np.vstack(out)

def main():
    smoke = sys.argv[1] == "smoke"
    from sklearn.svm import SVC
    d = common.load_raw()
    ntr = 3000 if smoke else 120000
    te0, te1, step = (120000, 121000, 500) if smoke else (120000, 220000, 20000)
    t0 = time.time()
    A = feats(d, 0, ntr); y = np.asarray(d[0:ntr, 0]).copy()
    m1 = sum(float(A[i:i+10000].astype(np.float64).sum()) for i in range(0, len(A), 10000)) / A.size
    m2 = sum(float((A[i:i+10000].astype(np.float64) ** 2).sum()) for i in range(0, len(A), 10000)) / A.size
    g = 1.0 / (A.shape[1] * (m2 - m1 * m1))
    A = A.astype(np.float64)  # float32 사본은 바로 버림 (7GB 한도)
    print(f"features {A.shape} {time.time()-t0:.0f}s gamma {g:.3e}", flush=True)
    t = time.time()
    m = SVC(C=10, gamma=g, cache_size=800).fit(A, y); del A
    fit_s = time.time() - t
    print(f"fit {fit_s:.0f}s SV {len(m.support_)}", flush=True)
    wr = 0; errs = []
    for s in range(te0, te1, step):
        B = feats(d, s, s + step); ys = np.asarray(d[s:s + step, 0])
        pr = m.predict(B.astype(np.float64)); w = int((pr != ys).sum()); wr += w
        errs += [(int(a), int(b)) for a, b in zip(ys[pr != ys], pr[pr != ys])]
        print(f"  chunk {s}: wrong {w}", flush=True)
    n = te1 - te0; tot = time.time() - t0
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    pairs = Counter(errs).most_common(5)
    line = (f"| 기준 SVC (C=10, g1.0) | {ntr:,} | {wr} / {n:,} | {1 - wr / n:.5f} | {fit_s:.0f}s | {tot:.0f}s | "
            f"{len(m.support_):,} | {mem}MB | {pairs} |")
    print("RESULT", line, flush=True)
    if not smoke:
        with open(OUT, "w") as f:
            f.write("| 모델 | 학습 개수 | 오답 / 채점 | 정확도 | 학습 시간 | 전체 시간 | 서포트 벡터 수 | 최대 메모리 | 오답쌍 상위 (정답,예측) |\n"
                    "|---|---|---|---|---|---|---|---|---|\n" + line + "\n"
                    "| (참고, 메인 big2) svc15 g1.2 | 120,000 | 349 / 100,000 | 0.99651 | | 3570s | | | |\n"
                    "| (참고, 메인 big2) svc15+HOG73 | 120,000 | 340 / 100,000 | 0.99660 | | 4217s | | | |\n")

if __name__ == "__main__":
    main()
