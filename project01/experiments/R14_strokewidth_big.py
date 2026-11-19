# R14: svc15+HOG73 에 '획 두께 정규화' 이미지의 HOG(7,3) 추가, 큰 데이터 조건 (19회차)
# 배경: 단일 SVC 특성 축(구멍·골격 끝점·사분면·보정 전 HOG)은 모두 효과 없음. 아직 안 해 본 축은 '펜 굵기'.
#       같은 9·7·4 라도 굵게 쓴 글씨는 고리가 메워지고 꺾임이 뭉개져 HOG 방향이 흐려진다.
#       각 이미지 하나에서 골격(1픽셀, skimage skeletonize)을 만든 뒤 3x3 십자로 한 번 팽창 → 모든 글씨가 같은 굵기(약 3픽셀).
#       이 이미지의 HOG(7,3)를 새 묶음으로 추가(W01 45쪽 특성 추출, W06 31쪽 모양 구조 특성). 데이터 개수는 그대로(증강 아님, 이미지를 대체·추가하지 않고 특성만 추가).
# 방법: big2·R12·R13 과 같은 구간(학습 0~12만, 채점 12만~22만), svc15+HOG73 g1.2 C=10 에 swH73 x1. 비교값 340 (R12 C=5 336, R13 350).
# 판정: 340 대비 30 이상 감소 유력, 50 이상 개선. 아니면 버림.
# 규칙: 원본 24만 안에서 구간만 나눔, PCA·StratifiedKFold 없음, 레이블은 학습에만 사용, 학습·채점 겹치지 않음. 허용 모델 SVC 만.
# 사용: python3 R14_strokewidth_big.py <sw73> [smoke]
import sys, os, json, time, resource
import numpy as np
from collections import Counter
import common
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "R14_strokewidth_big.md")
nb = json.load(open(os.path.join(HERE, "..", "mini_proj_01_20221329_stack.ipynb")))
src = ["".join(c["source"]) for c in nb["cells"]]
g = globals(); exec(src[3], g); exec(src[4].split("F = feat_svc15")[0], g)

def one_svc15_h73(X):
    dk = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
    return np.hstack([dk.reshape(len(dk), -1), hog(dk, 4, 2), hog(dk, 8, 4) * 2, hog(dk, 7, 3),
                      stroke_features(dk) * 1.5, profiles(dk)])

from scipy import ndimage
from skimage.morphology import skeletonize
_CROSS = ndimage.generate_binary_structure(2, 1)
def stroke_norm(dk):
    out = np.empty_like(dk)
    for i in range(len(dk)):
        sk = skeletonize(dk[i] > 0.3)
        out[i] = ndimage.binary_dilation(sk, _CROSS).astype(np.float32)
    return ndimage.gaussian_filter(out, sigma=(0, 0.5, 0.5))

def sw_h73(w):
    def f(X):
        dk = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
        return np.hstack([one_svc15_h73(X), hog(stroke_norm(dk), 7, 3) * w])
    return f

MODELS = {"sw73": (sw_h73(1.0), 1.2, 10)}
NAMES = {"sw73": "svc15+HOG73 + 획두께정규화 HOG73 x1"}

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
