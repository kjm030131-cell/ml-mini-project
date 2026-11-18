import json
st=json.load(open("/mnt/user-data/outputs/mini_proj_01_20221329_stack.ipynb"))
S=["".join(c["source"]) for c in st["cells"]]
cells=[S[0],S[1],S[2],S[3],
'''def make_features(X_raw, chunk=10000):
    out = np.empty((len(X_raw), 2801), dtype=np.float32)
    for s in range(0, len(X_raw), chunk):
        X = X_raw[s:s + chunk] / 255.0
        dk = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
        out[s:s + chunk] = np.hstack([dk.reshape(len(dk), -1), hog(dk, 4, 2), hog(dk, 8, 4) * 2,
                                      stroke_features(dk) * 1.5, profiles(dk)])
    return out

X_fe = make_features(X_train)
m1 = np.mean([X_fe[s:s + 10000].mean() for s in range(0, len(X_fe), 10000)])
m2 = np.mean([(X_fe[s:s + 10000].astype(np.float64) ** 2).mean() for s in range(0, len(X_fe), 10000)])
gamma_fe = 1.2 / (X_fe.shape[1] * (m2 - m1 ** 2))
X_fe.shape, gamma_fe''',
'''from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

Xs_raw, Xs, ys = X_train[:20000] / 255.0, X_fe[:20000], y_train[:20000]

print("SGD raw     ", cross_val_score(SGDClassifier(random_state=42), Xs_raw, ys, cv=3, scoring="accuracy").mean())
print("SVC raw     ", cross_val_score(SVC(C=10), Xs_raw, ys, cv=3, scoring="accuracy").mean())
print("SVC features", cross_val_score(SVC(C=10, gamma=gamma_fe), Xs, ys, cv=3, scoring="accuracy").mean())''',
'''final_clf = SVC(C=10, gamma=gamma_fe, cache_size=2000)

cv_res = cross_val_score(final_clf, X_fe, y_train, cv=10, scoring="accuracy", verbose=3)
cv_res''',
'''np.mean(cv_res)''']
nb={"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":c.splitlines(keepends=True)} for c in cells],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
json.dump(nb,open("/mnt/user-data/outputs/mini_proj_01_20221329_svc15.ipynb","w"),ensure_ascii=False,indent=1); print("ok")
