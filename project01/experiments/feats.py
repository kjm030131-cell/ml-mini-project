import numpy as np
def hog_features(imgs, cell=4, bins=9):
    gy, gx = np.gradient(imgs, axis=(1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins
    b = np.minimum(ang.astype(np.int32), bins - 1)
    n, h, w = imgs.shape
    ch, cw = h // cell, w // cell
    hist = np.zeros((n, ch, cw, bins), dtype=np.float32)
    for k in range(bins):
        hist[..., k] = (mag * (b == k)).reshape(n, ch, cell, cw, cell).sum(axis=(2, 4))
    norm = np.sqrt((hist ** 2).sum(axis=-1, keepdims=True)) + 1e-6
    return (hist / norm).reshape(n, -1)
def simple(X):
    imgs = X.reshape(-1, 28, 28)
    return np.hstack([imgs.sum(2), imgs.sum(1), imgs.reshape(-1,7,4,7,4).mean(axis=(2,4)).reshape(-1,49),
                      (imgs > 0.1).sum(axis=(1,2))[:, None].astype(np.float32)])
def build(Xraw, hog=True):
    X = Xraw.astype(np.float32) / 255.0
    parts=[X, simple(X)]
    if hog:
        imgs=X.reshape(-1,28,28); parts += [hog_features(imgs,4), hog_features(imgs,7)]
    return np.hstack(parts).astype(np.float32)
from scipy import ndimage
def holes(imgs, th=0.3):
    out = np.zeros((len(imgs), 2), dtype=np.float32)
    for i, im in enumerate(imgs):
        bg = im < th
        lab, n = ndimage.label(bg)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        inner = [k for k in range(1, n + 1) if k not in border]
        out[i, 0] = len(inner)
        out[i, 1] = sum((lab == k).sum() for k in inner)
    return out
def build2(Xraw):
    X = Xraw.astype(np.float32) / 255.0
    imgs = X.reshape(-1, 28, 28)
    return np.hstack([X, simple(X), hog_features(imgs, 4), hog_features(imgs, 7), holes(imgs)]).astype(np.float32)
def deskew(imgs):
    out = np.empty_like(imgs)
    yy, xx = np.mgrid[:28, :28].astype(np.float32)
    for i, im in enumerate(imgs):
        tot = im.sum() + 1e-6
        cy, cx = (yy * im).sum() / tot, (xx * im).sum() / tot
        vy = ((yy - cy) ** 2 * im).sum() / tot
        cov = ((yy - cy) * (xx - cx) * im).sum() / tot
        alpha = cov / (vy + 1e-6)
        mat = np.array([[1, 0], [alpha, 1]], dtype=np.float32)
        center = np.array([14, 14], dtype=np.float32)
        offset = np.array([cy, cx], dtype=np.float32) - mat @ center
        out[i] = ndimage.affine_transform(im, mat, offset=offset, order=1)
    return out
def build3(Xraw):
    X = Xraw.astype(np.float32) / 255.0
    imgs = X.reshape(-1, 28, 28)
    dk = deskew(np.ascontiguousarray(imgs.transpose(0, 2, 1)))
    return np.hstack([X, simple(X), hog_features(imgs, 4), hog_features(imgs, 7), holes(imgs),
                      dk.reshape(len(dk), -1), hog_features(dk, 4)]).astype(np.float32)
def hog2(imgs, cell=4, stride=2, bins=9):
    # 겹치는 칸 + 방향 구간 부드럽게 나누기
    gy, gx = np.gradient(imgs, axis=(1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins - 0.5
    lo = np.floor(ang).astype(np.int32); w = (ang - lo).astype(np.float32)
    n = len(imgs); pos = range(0, 28 - cell + 1, stride)
    hist = np.zeros((n, len(pos), len(pos), bins), dtype=np.float32)
    for k in range(bins):
        m = mag * ((lo % bins == k) * (1 - w) + ((lo + 1) % bins == k) * w)
        c = np.cumsum(np.cumsum(np.pad(m, ((0, 0), (1, 0), (1, 0))), axis=1), axis=2)
        for a, i in enumerate(pos):
            for b, j in enumerate(pos):
                hist[:, a, b, k] = c[:, i + cell, j + cell] - c[:, i, j + cell] - c[:, i + cell, j] + c[:, i, j]
    norm = np.sqrt((hist ** 2).sum(axis=-1, keepdims=True)) + 1e-6
    return (hist / norm).reshape(n, -1)
def size_norm(imgs, out=20):
    # 획이 있는 영역만 잘라서 20x20으로 맞춘 뒤 28x28 가운데에 배치(크기 정규화)
    res = np.zeros_like(imgs)
    for i, im in enumerate(imgs):
        r = np.where(im.max(1) > 0.1)[0]; c = np.where(im.max(0) > 0.1)[0]
        if len(r) == 0: continue
        box = im[r[0]:r[-1]+1, c[0]:c[-1]+1]
        s = out / max(box.shape)
        z = ndimage.zoom(box, s, order=1)
        h, w = z.shape; y0, x0 = (28 - h) // 2, (28 - w) // 2
        res[i, y0:y0+h, x0:x0+w] = np.clip(z, 0, 1)
    return res
def parts_all(Xraw, chunk=10000):
    P = {}
    for s in range(0, len(Xraw), chunk):
        X = Xraw[s:s+chunk].astype(np.float32)/255.0
        dk = deskew(np.ascontiguousarray(X.reshape(-1,28,28).transpose(0,2,1)))
        sn = size_norm(dk)
        q = dict(dkpix=dk.reshape(len(dk),-1), h42=hog2(dk,4,2), h84=hog2(dk,8,4),
                 h42b12=hog2(dk,4,2,12), snpix=sn.reshape(len(sn),-1), sn42=hog2(sn,4,2))
        for k,v in q.items(): P.setdefault(k, []).append(v.astype(np.float32))
    return {k: np.vstack(v) for k,v in P.items()}

# --- 질감 특성 (실험 F) ---
def lbp_feat(imgs, cell=7):
    # 각 픽셀이 이웃 8개 중 몇 개보다 밝은지(0~8)를 7x7 칸마다 히스토그램 → 16칸 x 9 = 144
    p = np.pad(imgs, ((0, 0), (1, 1), (1, 1)))
    c = imgs
    cnt = np.zeros_like(imgs, dtype=np.int8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            cnt += (p[:, 1 + dy:29 + dy, 1 + dx:29 + dx] >= c).astype(np.int8)
    cnt = np.where(c > 0.05, cnt, -1)          # 배경은 제외
    n = len(imgs); g = 28 // cell
    out = np.zeros((n, g, g, 9), np.float32)
    for k in range(9):
        out[..., k] = (cnt == k).reshape(n, g, cell, g, cell).sum(axis=(2, 4))
    tot = out.sum(-1, keepdims=True) + 1e-6
    return (out / tot).reshape(n, -1)

def gabor_feat(imgs, cell=7):
    # 4방향 x 2주파수 가보르 필터 반응 크기를 7x7 칸마다 평균 → 8 x 16 = 128
    feats = []
    yy, xx = np.mgrid[-4:5, -4:5].astype(np.float32)
    for lam in (4.0, 7.0):
        for th in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            xr = xx * np.cos(th) + yy * np.sin(th); yr = -xx * np.sin(th) + yy * np.cos(th)
            k = np.exp(-(xr ** 2 + yr ** 2) / (2 * (lam / 2) ** 2)) * np.cos(2 * np.pi * xr / lam)
            k -= k.mean()
            r = np.abs(ndimage.convolve(imgs, k[None], mode="constant"))
            n = len(imgs); g = 28 // cell
            feats.append(r.reshape(n, g, cell, g, cell).mean(axis=(2, 4)).reshape(n, -1))
    f = np.hstack(feats)
    return f / (f.max(axis=1, keepdims=True) + 1e-6)

# --- HOG 계산 방식 개선 (실험 G) ---
def _grad(imgs, sigma=0.0):
    if sigma > 0: imgs = ndimage.gaussian_filter(imgs, (0, sigma, sigma))
    gy, gx = np.gradient(imgs, axis=(1, 2))
    return gx, gy

def hog3(imgs, cell=4, stride=2, bins=9, signed=False, sigma=0.0):
    # hog2 + 부호 있는 방향(0~360도) 옵션 + 계산 전 가우시안 흐림 옵션
    gx, gy = _grad(imgs, sigma)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    rng = 2 * np.pi if signed else np.pi
    ang = (np.arctan2(gy, gx) % rng) / rng * bins - 0.5
    lo = np.floor(ang).astype(np.int32); w = (ang - lo).astype(np.float32)
    n = len(imgs); pos = range(0, 28 - cell + 1, stride)
    hist = np.zeros((n, len(pos), len(pos), bins), dtype=np.float32)
    for k in range(bins):
        m = mag * ((lo % bins == k) * (1 - w) + ((lo + 1) % bins == k) * w)
        c = np.cumsum(np.cumsum(np.pad(m, ((0, 0), (1, 0), (1, 0))), axis=1), axis=2)
        for a, i in enumerate(pos):
            for b, j in enumerate(pos):
                hist[:, a, b, k] = c[:, i + cell, j + cell] - c[:, i, j + cell] - c[:, i + cell, j] + c[:, i, j]
    norm = np.sqrt((hist ** 2).sum(axis=-1, keepdims=True)) + 1e-6
    return (hist / norm).reshape(n, -1)

def hog_dt(imgs, cell=4, bins=9, sigma=0.0):
    # Dalal-Triggs 방식: 겹치지 않는 칸 히스토그램 → 2x2 칸 묶음(블록)마다 L2-Hys 정규화
    gx, gy = _grad(imgs, sigma)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins - 0.5
    lo = np.floor(ang).astype(np.int32); w = (ang - lo).astype(np.float32)
    n = len(imgs); g = 28 // cell; s = g * cell
    hist = np.zeros((n, g, g, bins), np.float32)
    for k in range(bins):
        m = mag * ((lo % bins == k) * (1 - w) + ((lo + 1) % bins == k) * w)
        hist[..., k] = m[:, :s, :s].reshape(n, g, cell, g, cell).sum(axis=(2, 4))
    blocks = []
    for i in range(g - 1):
        for j in range(g - 1):
            v = hist[:, i:i + 2, j:j + 2, :].reshape(n, -1)
            v = v / (np.sqrt((v ** 2).sum(1, keepdims=True)) + 1e-6)
            v = np.minimum(v, 0.2)
            v = v / (np.sqrt((v ** 2).sum(1, keepdims=True)) + 1e-6)
            blocks.append(v)
    return np.hstack(blocks)
