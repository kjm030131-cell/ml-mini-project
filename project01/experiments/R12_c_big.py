# R12: svc15+HOG73 의 C 재조정, 큰 데이터 조건 (17회차)
# 배경: 12만/10만(학습 0~12만, 채점 12만~22만)에서 기준 366 / svc15 349 / svc15+HOG73(C=10, g1.2) 340 (R10·big2).
#       C·gamma 는 5만·10만 학습 2만 채점에서 평탄했지만, 2만 채점은 ±9 우연 범위라 못 가렸다. 10만 채점으로 C 5·20 을 짝 비교.
# 방법: big2 와 같은 구간·특성(svc15+HOG73, gamma 1.2배)에서 C 만 5, 20 으로 바꿈. C=10 값은 big2 의 340.
# 판정: 340 대비 30 이상 감소 '유력', 50 이상 '개선'. 아니면 C=10 유지.
# 규칙: 원본 24만 안에서 구간만 나눔(증강 아님), PCA·StratifiedKFold 없음, 레이블은 학습에만 사용, 학습·채점 겹치지 않음. 허용 모델 SVC 만.
# 사용: python3 R12_c_big.py <c5|c20> [smoke]
import sys, os, json, time, resource
import numpy as np
from collections import Counter
import common
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "R12_c_big.md")
nb = json.load(open(os.path.join(HERE, "..", "mini_proj_01_20221329_stack.ipynb")))
src = ["".join(c["source"]) for c in nb["cells"]]
g = globals(); exec(src[3], g); exec(src[4].split("F = feat_svc15")[0], g)

def one_svc15_h73(X):
    dk = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
    return np.hstack([dk.reshape(len(dk), -1), hog(dk, 4, 2), hog(dk, 8, 4) * 2, hog(dk, 7, 3),
                      stroke_features(dk) * 1.5, profiles(dk)])

MODELS = {"c5": (one_svc15_h73, 1.2, 5), "c20": (one_svc15_h73, 1.2, 20)}
NAMES = {"c5": "svc15+HOG73 g1.2 C=5", "c20": "svc15+HOG73 g1.2 C=20"}

def main():
    key = sys.argv[1]; smoke = len(sys.argv) > 2
    f, gm, Cv = MODELS[key]
    from sklearn.svm import SVC
    d = common.load_raw()
    tr0, tr1 = (0, 3000) if smoke else (0, 120000)
    te0, te1, step = (120000, 121000, 500) if smoke else (120000, 220000, 20000)
    t0 = time.time()
    A = by_chunk(f, np.asarray(d[tr0:tr1, 1:]).astype(np.float32)); y = np.asarray(d[tr0:tr1, 0]).copy()
    m1 = sum(A[i:i+10000].sum() for i in range(0, len(A), 10000)) / A.size
    m2 = sum((A[i:i+10000] ** 2).sum() for i in range(0, len(A), 10000)) / A.size
    gamma = gm / (A.shape[1] * (m2 - m1 * m1))
    print(f"{key} features {A.shape} {time.time()-t0:.0f}s gamma {gamma:.3e}", flush=True)
    t = time.time(); m = SVC(C=Cv, gamma=gamma, cache_size=600).fit(A, y); del A
    fit_s = time.time() - t; print(f"{key} fit {fit_s:.0f}s SV {len(m.support_)}", flush=True)
    wr = 0; errs = []; chunks = []
    for s in range(te0, te1, step):
        B = by_chunk(f, np.asarray(d[s:s+step, 1:]).astype(np.float32)); ys = np.asarray(d[s:s+step, 0])
        pr = m.predict(B); w = int((pr != ys).sum()); wr += w; chunks.append(w)
        errs += [(int(a), int(b)) for a, b in zip(ys[pr != ys], pr[pr != ys])]
        print(f"  {key} chunk {s}: wrong {w}", flush=True)
    n = te1 - te0; tot = time.time() - t0
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    line = (f"| {NAMES[key]} | {wr} / {n:,} | {1 - wr / n:.5f} | {chunks} | {fit_s:.0f}s | {tot:.0f}s | {mem}MB | "
            f"{Counter(errs).most_common(5)} |")
    print("RESULT", line, flush=True)
    if not smoke:
        new = not os.path.exists(OUT)
        with open(OUT, "a") as fo:
            if new:
                fo.write("| 모델 | 오답 / 채점 | 정확도 | 2만 구간별 오답 | 학습 시간 | 전체 시간 | 최대 메모리 | 오답쌍 상위 (정답,예측) |\n"
                         "|---|---|---|---|---|---|---|---|\n")
            fo.write(line + "\n")

if __name__ == "__main__":
    main()
