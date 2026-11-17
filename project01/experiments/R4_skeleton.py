# R4: 골격(skeleton) 끝점·갈림점 위치 특성 — 3↔5 등 오답쌍 겨냥 (대기열 0-1, 2)
# 배경: 15차 단일 SVC(오답 88)와 Stacking(78~81) 모두 5→3 이 오답 1위. 윤곽선·구멍·전환 횟수로는 안 줄었다.
#       3 은 왼쪽에 끝점 3개(위·가운데·아래)와 오른쪽 가운데 갈림점, 5 는 오른쪽 위 끝점 + 왼쪽 위 꺾임 + 왼쪽 아래 끝점.
#       → 이미지 하나에서 획을 1픽셀 두께 골격으로 줄이고(skimage.morphology.skeletonize),
#         이웃 1개인 골격 픽셀(끝점)·3개 이상(갈림점)의 개수와 3x3 영역별 위치 히스토그램을 특성으로 쓴다 (총 20개).
#       레이블 미사용, 데이터 개수 그대로 → 규칙(특성 추출) 안.
# 사용: python3 R4_skeleton.py base | sk15 | sk3 | smoke
import sys, time, os
import numpy as np
from collections import Counter
from scipy import ndimage
from skimage.morphology import skeletonize
from common import feat, labels, ROOT, CACHE
from feats import holes

OUT = os.path.join(ROOT, "..", "results", "R4_skeleton.md")

def img(p): return feat(p, "dkpix").reshape(-1, 28, 28)

def extra14(p):  # exp15.py 와 동일
    im = img(p); b = (im > 0.3).astype(np.int8)
    tr_r = (np.diff(b, axis=2) == 1).sum(2) / 4.0; tr_c = (np.diff(b, axis=1) == 1).sum(1) / 4.0
    s = im.sum((1, 2)).clip(1e-6)
    top = im[:, :10, :].sum((1, 2)) / s; left = im[:, :, :14].sum((1, 2)) / s
    h = holes(im) / np.array([1, 50], np.float32)
    return np.hstack([tr_r, tr_c, top[:, None], left[:, None], h]).astype(np.float32)

def profiles(p):  # exp15.py 와 동일
    b = img(p) > 0.3
    def first(m, axis):
        has = m.any(axis); f = np.where(has, np.argmax(m, axis), 28); return f / 28.0
    left = first(b, 2); right = 1 - first(b[:, :, ::-1], 2); top = first(b, 1); bottom = 1 - first(b[:, ::-1, :], 1)
    return np.hstack([left, right, top, bottom]).astype(np.float32)

K = np.ones((3, 3), np.int32); K[1, 1] = 0
EDGES = [0, 9, 19, 28]
def skel_feats(p):
    cp = os.path.join(CACHE, f"{p}_skel.npy")
    if os.path.exists(cp): return np.load(cp)
    im = img(p); out = np.zeros((len(im), 20), np.float32)
    for i, x in enumerate(im):
        sk = skeletonize(x > 0.3)
        nb = ndimage.convolve(sk.astype(np.int32), K, mode="constant") * sk
        ends = (nb == 1); junc = (nb >= 3)
        out[i, 0] = ends.sum() / 4.0; out[i, 1] = min(junc.sum(), 8) / 4.0
        for a in range(3):
            for c in range(3):
                r = slice(EDGES[a], EDGES[a + 1]); q = slice(EDGES[c], EDGES[c + 1])
                out[i, 2 + a * 3 + c] = min(ends[r, q].sum(), 2) / 2.0
                out[i, 11 + a * 3 + c] = min(junc[r, q].sum(), 2) / 2.0
    np.save(cp, out); return out

def base(p): return np.hstack([feat(p, "dkpix"), feat(p, "h42"), feat(p, "h84") * 2, extra14(p) * 1.5, profiles(p)])

if __name__ == "__main__":
    mode = sys.argv[1]
    from sklearn.svm import SVC
    y, yte = labels()
    t0 = time.time()
    A = base("tr"); B = base("te")
    w = {"base": 0, "sk15": 1.5, "sk3": 3.0, "smoke": 1.5}[mode]
    if w:
        A = np.hstack([A, skel_feats("tr") * w]); B = np.hstack([B, skel_feats("te") * w])
    A = A.astype(np.float64); B = B.astype(np.float64)
    if mode == "smoke":
        A, y, B, yte = A[:2000], y[:2000], B[:500], yte[:500]
    print(f"features {A.shape} ready in {time.time()-t0:.0f}s", flush=True)
    t = time.time()
    m = SVC(C=10, gamma=1.2 / (A.shape[1] * A.var()), cache_size=500).fit(A, y)
    pr = m.predict(B); wr = int((pr != yte).sum())
    pairs = Counter((int(a), int(b)) for a, b in zip(yte[pr != yte], pr[pr != yte])).most_common(5)
    line = f"| {mode} (골격 x{w}) | {wr} | {1 - wr / len(yte):.4f} | {time.time() - t:.0f}s | {pairs} |"
    print("RESULT", line, flush=True)
    if mode != "smoke":
        new = not os.path.exists(OUT)
        with open(OUT, "a") as f:
            if new:
                f.write("| 실험 | 오답 (2만 중) | 정확도 | 시간 | 오답쌍 상위 (정답,예측) |\n|---|---|---|---|---|\n")
            f.write(line + "\n")
