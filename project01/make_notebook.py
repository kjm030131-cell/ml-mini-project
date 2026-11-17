import json
cells = [
'''import numpy as np
import pandas as pd

input_file = "emnist-digits-train.csv"
df_train = pd.read_csv(input_file, header=None)''',
'''data_train = df_train.to_numpy()
data_train.shape''',
'''X_train = data_train[:, 1:]
y_train = data_train[:, 0]
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

def make_features(X_raw, chunk=10000):
    out = []
    for s in range(0, len(X_raw), chunk):
        X = X_raw[s:s + chunk].astype(np.float32) / 255.0
        imgs = np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1))
        dk = deskew(imgs)
        out.append(np.hstack([dk.reshape(len(dk), -1), hog(dk, 4, 2), hog(dk, 8, 4)]).astype(np.float32))
    return np.vstack(out)

X_fe = make_features(X_train)
X_fe.shape''',
'''from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

Xs_raw, Xs, ys = X_train[:20000] / 255.0, X_fe[:20000], y_train[:20000]

print("SGD raw     ", cross_val_score(SGDClassifier(random_state=42), Xs_raw, ys, cv=3, scoring="accuracy").mean())
print("SVC raw     ", cross_val_score(SVC(C=10), Xs_raw, ys, cv=3, scoring="accuracy").mean())
print("SVC features", cross_val_score(SVC(C=10), Xs, ys, cv=3, scoring="accuracy").mean())''',
'''final_clf = SVC(C=10, cache_size=2000)

cv_res = cross_val_score(final_clf, X_fe, y_train, cv=10, scoring="accuracy", verbose=3)
cv_res''',
'''np.mean(cv_res)''',
'''from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_pred_s = cross_val_predict(SVC(C=10), X_fe[:30000], y_train[:30000], cv=10)
conf_mx = confusion_matrix(y_train[:30000], y_pred_s)
conf_mx''',
'''row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
np.round(norm_conf_mx, 4)''',
]
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": c.splitlines(keepends=True)} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("/home/claude/mini_proj_01_20221329.ipynb", "w"), ensure_ascii=False, indent=1)
