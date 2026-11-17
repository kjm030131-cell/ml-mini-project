# 병렬 실험 세션 공통 코드: 데이터 로드, 특성 캐시, SVC 평가
# 기준: 학습 = 앞 5만 개, 채점(홀드아웃) = 220000~240000 (2만 개). 기준 SVC 오답 100개.
import numpy as np, os, time, glob
from feats import deskew, hog2, holes
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "..", "data")
CACHE = os.path.join(ROOT, "cache"); os.makedirs(CACHE, exist_ok=True)
def load_raw():
    p = os.path.join(CACHE, "emnist.npy")
    if not os.path.exists(p):
        parts = sorted(glob.glob(os.path.join(DATA, "emnist_digits_train_part*.npz")))
        np.save(p, np.vstack([np.load(f)["d"] for f in parts]))
    return np.load(p, mmap_mode="r")
TR, TE = slice(0, 50000), slice(220000, 240000)
def labels():
    d = load_raw(); return np.asarray(d[TR, 0]), np.asarray(d[TE, 0])
def _imgs(part, deskewed=True):
    d = load_raw(); sl = TR if part == "tr" else TE
    X = np.asarray(d[sl, 1:]).astype(np.float32) / 255.0
    im = np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1))
    return deskew(im) if deskewed else im
def feat(part, key):
    """key: dkpix, h42, h84, holes, rawpix, rh42, rows, cols"""
    p = os.path.join(CACHE, f"{part}_{key}.npy")
    if not os.path.exists(p):
        if key in ("rawpix", "rh42", "holes"): im = _imgs(part, False)
        else: im = _imgs(part, True)
        f = {"dkpix": lambda: im.reshape(len(im), -1), "rawpix": lambda: im.reshape(len(im), -1),
             "h42": lambda: hog2(im, 4, 2), "h84": lambda: hog2(im, 8, 4), "rh42": lambda: hog2(im, 4, 2),
             "h73": lambda: hog2(im, 7, 3), "h62": lambda: hog2(im, 6, 2),
             "holes": lambda: holes(im) / np.array([1, 50], np.float32),
             "rows": lambda: im.sum(2) / 28, "cols": lambda: im.sum(1) / 28}[key]()
        np.save(p, f.astype(np.float32))
    return np.load(p)
def block(part, spec):
    """spec: [(key, weight), ...] -> float64 행렬"""
    return np.hstack([feat(part, k) * w for k, w in spec]).astype(np.float64)
def run_svc(name, spec, log, C=10, gamma_mult=1.0, **kw):
    from sklearn.svm import SVC
    y, yte = labels(); A = block("tr", spec); B = block("te", spec)
    g = gamma_mult / (A.shape[1] * A.var())
    t = time.time(); m = SVC(C=C, gamma=g, cache_size=500, **kw).fit(A, y)
    w = int((m.predict(B) != yte).sum())
    line = f"| {name} | {w} | {(1 - w / len(yte)):.4f} | {time.time() - t:.0f}s |"
    print(line, flush=True); open(log, "a").write(line + "\n")
    return w
