# R10: 기준 SVC 를 메인 세션의 '학습 12만 / 채점 10만' 방식으로 (15회차)
# 배경: 메인 세션 big2.py 가 채점 구간을 2만 → 10만(120000~220000)으로 넓혀 svc15 349, +HOG73 340, HOG12방향 380 을 얻음.
#       2만 채점은 ±9 가 우연 범위라 10만 채점이 훨씬 정확하다(오답 350 의 표준오차 약 ±19, 차이의 표준오차 약 ±12 미만—같은 문제를 푸니 더 작음).
#       그런데 이 방식에 '기준 SVC'(제출 노트북 조합, 24만 10-fold 0.9964125) 값이 없어 svc15 가 정말 나은지 판정할 수 없다.
# 방법: 학습 0~120000, 채점 120000~220000 (2만씩 5구간). 기준 조합(보정 픽셀 + HOG42 + HOG84, C=10, gamma 1.0) 1회.
#       big2 svc15(349)와 같은 문제를 풀므로 오답 차이를 바로 비교. 구간별 오답도 기록.
# 규칙: 원본 24만 안의 구간만 사용(증강 아님), PCA·StratifiedKFold 없음, 학습·채점 구간 겹치지 않음, CPU 만.
# 사용: python3 R10_base_big.py full   /   python3 R10_base_big.py smoke
import sys, os, time, resource
import numpy as np
from collections import Counter
import common
from feats import deskew, hog2
OUT = os.path.join(common.ROOT, "..", "results", "R10_base_big.md")

def featurize(X):
    im = np.ascontiguousarray((X.astype(np.float32) / 255.0).reshape(-1, 28, 28).transpose(0, 2, 1))
    im = deskew(im)
    return np.hstack([im.reshape(len(im), -1), hog2(im, 4, 2), hog2(im, 8, 4)]).astype(np.float64)

def by_chunk(d, a, b, step=20000):
    return np.vstack([featurize(np.asarray(d[s:min(s + step, b), 1:])) for s in range(a, b, step)])

def main():
    smoke = sys.argv[1] == "smoke"
    ntr, te0, te1, step = (3000, 120000, 121000, 500) if smoke else (120000, 120000, 220000, 20000)
    from sklearn.svm import SVC
    d = common.load_raw()
    t0 = time.time()
    A = by_chunk(d, 0, ntr); y = np.asarray(d[0:ntr, 0])
    print(f"train features {A.shape} {time.time()-t0:.0f}s", flush=True)
    m = SVC(C=10, gamma=1.0 / (A.shape[1] * A.var()), cache_size=1000).fit(A, y)
    fit_s = time.time() - t0; nsv = len(m.support_); del A
    print(f"fit {fit_s:.0f}s  SV {nsv}", flush=True)
    per, wrong_pairs, tot = [], Counter(), 0
    for s in range(te0, te1, step):
        B = featurize(np.asarray(d[s:s + step, 1:])); yt = np.asarray(d[s:s + step, 0])
        pr = m.predict(B); w = int((pr != yt).sum()); per.append(w); tot += len(yt)
        wrong_pairs.update((int(a), int(b)) for a, b in zip(yt[pr != yt], pr[pr != yt]))
        print(f"chunk {s} wrong {w}", flush=True)
    W = sum(per); el = time.time() - t0
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    line = (f"| 기준 SVC (C=10, g1.0) | {ntr:,} | {W} / {tot:,} | {1 - W / tot:.5f} | {per} | {fit_s:.0f}s | {el:.0f}s | "
            f"{nsv:,} | {mem}MB | {wrong_pairs.most_common(5)} |")
    print("RESULT", line, flush=True)
    if not smoke:
        new = not os.path.exists(OUT)
        with open(OUT, "a") as f:
            if new:
                f.write("| 모델 | 학습 개수 | 오답 | 정확도 | 2만 구간별 오답 | 학습 시간 | 전체 | 서포트 벡터 | 최대 메모리 | 오답쌍 상위 (정답,예측) |\n"
                        "|---|---|---|---|---|---|---|---|---|---|\n")
            f.write(line + "\n")

if __name__ == "__main__":
    main()
