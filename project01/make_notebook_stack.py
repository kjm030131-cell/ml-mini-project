import json, sys
csv = sys.argv[1] if len(sys.argv) > 1 else "emnist-digits-train.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs/mini_proj_01_20221329_stack.ipynb"
cells = [
f'''import numpy as np
import pandas as pd

input_file = "{csv}"
df_train = pd.read_csv(input_file, header=None)''',
'''data_train = df_train.to_numpy()
data_train.shape''',
'''X_train = data_train[:, 1:].astype(np.float32)
y_train = data_train[:, 0].copy()
del df_train, data_train
X_train.shape, y_train.shape''',
'''from scipy import ndimage

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
        offset = np.array([cy, cx], dtype=np.float32) - mat @ np.array([14, 14], dtype=np.float32)
        out[i] = ndimage.affine_transform(im, mat, offset=offset, order=1)
    return out

def hog(imgs, cell, stride, bins=9):
    gy, gx = np.gradient(imgs, axis=(1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins - 0.5
    lo = np.floor(ang).astype(np.int32)
    w = (ang - lo).astype(np.float32)
    n = len(imgs)
    pos = range(0, 28 - cell + 1, stride)
    hist = np.zeros((n, len(pos), len(pos), bins), dtype=np.float32)
    for k in range(bins):
        m = mag * ((lo % bins == k) * (1 - w) + ((lo + 1) % bins == k) * w)
        c = np.cumsum(np.cumsum(np.pad(m, ((0, 0), (1, 0), (1, 0))), axis=1), axis=2)
        for a, i in enumerate(pos):
            for b, j in enumerate(pos):
                hist[:, a, b, k] = c[:, i + cell, j + cell] - c[:, i, j + cell] - c[:, i + cell, j] + c[:, i, j]
    norm = np.sqrt((hist ** 2).sum(axis=-1, keepdims=True)) + 1e-6
    return (hist / norm).reshape(n, -1)

def hog_block(imgs, cell=4, bins=9):
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

def holes(imgs, th=0.3):
    out = np.zeros((len(imgs), 2), dtype=np.float32)
    for i, im in enumerate(imgs):
        lab, n = ndimage.label(im < th)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        inner = [k for k in range(1, n + 1) if k not in border]
        out[i, 0] = len(inner)
        out[i, 1] = sum((lab == k).sum() for k in inner)
    return out

def simple(X):
    imgs = X.reshape(-1, 28, 28)
    return np.hstack([imgs.sum(2), imgs.sum(1), imgs.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(-1, 49),
                      (imgs > 0.1).sum(axis=(1, 2))[:, None].astype(np.float32)])

def stroke_features(dk):
    b = (dk > 0.3).astype(np.int8)
    tr_r = (np.diff(b, axis=2) == 1).sum(2) / 4.0
    tr_c = (np.diff(b, axis=1) == 1).sum(1) / 4.0
    ink = dk.sum((1, 2)).clip(1e-6)
    top = dk[:, :10, :].sum((1, 2)) / ink
    left = dk[:, :, :14].sum((1, 2)) / ink
    h = holes(dk) / np.array([1, 50], np.float32)
    return np.hstack([tr_r, tr_c, top[:, None], left[:, None], h]).astype(np.float32)

def profiles(dk):
    b = dk > 0.3
    def first(m, axis):
        return np.where(m.any(axis), np.argmax(m, axis), 28) / 28.0
    left = first(b, 2)
    right = 1 - first(b[:, :, ::-1], 2)
    top = first(b, 1)
    bottom = 1 - first(b[:, ::-1, :], 1)
    return np.hstack([left, right, top, bottom]).astype(np.float32)''',
'''def by_chunk(f, X_raw, dtype=np.float64, chunk=10000):
    first = f(X_raw[:chunk] / 255.0)
    out = np.empty((len(X_raw), first.shape[1]), dtype=dtype)
    out[:len(first)] = first
    for s in range(chunk, len(X_raw), chunk):
        out[s:s + chunk] = f(X_raw[s:s + chunk] / 255.0)
    return out

def one_svc15(X):
    dk = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
    return np.hstack([dk.reshape(len(dk), -1), hog(dk, 4, 2), hog(dk, 8, 4) * 2,
                      stroke_features(dk) * 1.5, profiles(dk)])

def one_lr(X):
    dk = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
    return np.hstack([dk.reshape(len(dk), -1), hog(dk, 4, 2), hog(dk, 8, 4)])

def one_raw(X):
    im = np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1))
    return np.hstack([im.reshape(len(im), -1), hog(im, 4, 2), hog(im, 8, 4)])

def one_lgbm(X):
    imgs = X.reshape(-1, 28, 28)
    dk = deskew(np.ascontiguousarray(imgs.transpose(0, 2, 1)))
    return np.hstack([X, simple(X), hog_block(imgs, 4), hog_block(imgs, 7), holes(imgs),
                      dk.reshape(len(dk), -1), hog_block(dk, 4)])

def feat_svc15(X): return by_chunk(one_svc15, X)
def feat_lr(X): return by_chunk(one_lr, X)
def feat_raw(X): return by_chunk(one_raw, X)
def feat_lgbm(X): return by_chunk(one_lgbm, X, np.float32)

F = feat_svc15(X_train[:50000])
gamma_svc15 = 1.2 / (F.shape[1] * F.var())
print(F.shape, feat_lr(X_train[:5]).shape, feat_raw(X_train[:5]).shape, feat_lgbm(X_train[:5]).shape, gamma_svc15)
del F''',
'''from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier

svc15_clf = make_pipeline(FunctionTransformer(feat_svc15), SVC(C=10, gamma=gamma_svc15, cache_size=500))
lgbm_clf = make_pipeline(FunctionTransformer(feat_lgbm),
                         LGBMClassifier(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3,
                                        n_jobs=4, random_state=42, verbose=-1))
raw_clf = make_pipeline(FunctionTransformer(feat_raw), SVC(C=10, cache_size=500))
lr_clf = make_pipeline(FunctionTransformer(feat_lr), LogisticRegression(C=10, max_iter=300))

stack_clf = StackingClassifier(
    estimators=[("svc15", svc15_clf), ("lgbm", lgbm_clf), ("raw", raw_clf), ("lr", lr_clf)],
    final_estimator=LogisticRegression(C=1, max_iter=4000),
    cv=2)''',
'''from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

Xs, ys = X_train[:20000], y_train[:20000]

print("SGD raw      ", cross_val_score(SGDClassifier(random_state=42), Xs / 255.0, ys, cv=3, scoring="accuracy").mean())
print("SVC raw      ", cross_val_score(SVC(C=10), Xs / 255.0, ys, cv=3, scoring="accuracy").mean())
print("SVC features ", cross_val_score(svc15_clf, Xs, ys, cv=3, scoring="accuracy").mean())''',
'''cv_res = cross_val_score(stack_clf, X_train, y_train, cv=10, scoring="accuracy", verbose=3, n_jobs=2)
cv_res''',
'''np.mean(cv_res)''',
]
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": c.splitlines(keepends=True)} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
