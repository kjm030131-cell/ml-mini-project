# R7: 3↔5 겨냥 '사분면 획 연결' 특성 (대기열 0-1 남은 후보: 왼쪽 위 획 방향, 위쪽 가로획 길이, 중간 높이 왼쪽 획)
# 배경: 15차 단일 SVC(88)·Stacking(78~81)·R4 골격(85) 모두 5→3 이 오답 1위.
#   5 = 위 가로획 + 왼쪽 위 '세로' 획(위에서 가운데로 내려옴) + 오른쪽 위 끝이 열림
#   3 = 위쪽 호가 오른쪽으로 휘어 가운데까지 내려옴 + 왼쪽 위는 짧은 끝만 있음
#   → 이미지 하나에서 이진화한 획을 영역(왼위/오른위/왼아래/오른아래, 서로 겹치게)으로 잘라
#     각 영역 안 가장 큰 연결 성분의 세로 길이·가로 길이·잉크 비율, 그리고 위쪽 1/3 의 가장 긴 가로 연속 구간
#     (길이·시작·끝 x)을 특성으로 쓴다 (총 4x3 + 3 + 2 = 17개).
#   레이블 미사용, 데이터 개수 그대로 → 규칙(특성 추출) 안. 기본 특성은 R4 base(15차, 오답 88) 그대로.
# 사용: python3 R7_quadrant.py q15 | q3 | q15sk | smoke
import sys, time, os
import numpy as np
from collections import Counter
from scipy import ndimage
from common import labels, ROOT, CACHE
from R4_skeleton import base, img, skel_feats

OUT = os.path.join(ROOT, "..", "results", "R7_quadrant.md")
REG = [(slice(0, 16), slice(0, 15)), (slice(0, 16), slice(13, 28)),
       (slice(12, 28), slice(0, 15)), (slice(12, 28), slice(13, 28))]

def quad_feats(p):
    cp = os.path.join(CACHE, f"{p}_quad.npy")
    if os.path.exists(cp): return np.load(cp)
    im = img(p); out = np.zeros((len(im), 17), np.float32)
    for i, x in enumerate(im):
        b = x > 0.3
        for k, (r, c) in enumerate(REG):
            sub = b[r, c]; lab, n = ndimage.label(sub)
            if n:
                sizes = np.bincount(lab.ravel())[1:]; m = lab == (np.argmax(sizes) + 1)
                ys, xs = np.nonzero(m)
                out[i, k * 3] = (ys.max() - ys.min() + 1) / sub.shape[0]
                out[i, k * 3 + 1] = (xs.max() - xs.min() + 1) / sub.shape[1]
                out[i, k * 3 + 2] = min(sub.mean() * 4, 1.0)
        best = (0, 0, 0)  # 위쪽 1/3 가장 긴 가로 연속 구간
        for row in b[:10]:
            d = np.diff(np.r_[0, row.astype(np.int8), 0]); st = np.nonzero(d == 1)[0]; en = np.nonzero(d == -1)[0]
            if len(st):
                j = np.argmax(en - st)
                if en[j] - st[j] > best[0]: best = (en[j] - st[j], st[j], en[j])
        out[i, 12:15] = np.array(best, np.float32) / 28.0
        # 중간 높이(10~16행) 왼쪽 1/3 잉크 유무, 오른쪽 1/3 잉크 유무
        out[i, 15] = float(b[10:17, :9].any()); out[i, 16] = float(b[10:17, 19:].any())
    np.save(cp, out); return out

if __name__ == "__main__":
    mode = sys.argv[1]
    from sklearn.svm import SVC
    y, yte = labels(); t0 = time.time()
    A = base("tr"); B = base("te")
    w = {"q15": 1.5, "q3": 3.0, "q15sk": 1.5, "smoke": 1.5}[mode]
    A = np.hstack([A, quad_feats("tr") * w]); B = np.hstack([B, quad_feats("te") * w])
    if mode == "q15sk":
        A = np.hstack([A, skel_feats("tr") * 1.5]); B = np.hstack([B, skel_feats("te") * 1.5])
    A = A.astype(np.float64); B = B.astype(np.float64)
    if mode == "smoke":
        A, y, B, yte = A[:2000], y[:2000], B[:500], yte[:500]
    print(f"features {A.shape} ready in {time.time()-t0:.0f}s", flush=True)
    t = time.time()
    m = SVC(C=10, gamma=1.2 / (A.shape[1] * A.var()), cache_size=500).fit(A, y)
    pr = m.predict(B); wr = int((pr != yte).sum())
    pairs = Counter((int(a), int(b)) for a, b in zip(yte[pr != yte], pr[pr != yte])).most_common(5)
    n53 = int(((yte == 5) & (pr == 3)).sum()); n35 = int(((yte == 3) & (pr == 5)).sum())
    line = f"| {mode} (사분면 x{w}) | {wr} | {1 - wr / len(yte):.4f} | {time.time() - t:.0f}s | 5→3 {n53}, 3→5 {n35} | {pairs} |"
    print("RESULT", line, flush=True)
    if mode != "smoke":
        new = not os.path.exists(OUT)
        with open(OUT, "a") as f:
            if new:
                f.write("| 실험 | 오답 (2만 중) | 정확도 | 시간 | 3↔5 | 오답쌍 상위 (정답,예측) |\n|---|---|---|---|---|---|\n")
            f.write(line + "\n")
