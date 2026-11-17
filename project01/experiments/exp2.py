import numpy as np, time
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from feats import *
d=np.load("emnist.npy"); N=20000
X, y = d[:N,1:], d[:N,0]
Xh=build2(X)
print("features", Xh.shape, flush=True)
def lg(**k):
    p=dict(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3, n_jobs=2, random_state=42, verbose=-1); p.update(k)
    return LGBMClassifier(**p)
for name, est in [("HOG+holes 300", lg()),
                  ("HOG+holes 600 nl63", lg(n_estimators=600, num_leaves=63, subsample=0.8, subsample_freq=1)),
                  ("HOG+holes 1000 lr0.05", lg(n_estimators=1000, learning_rate=0.05, subsample=0.8, subsample_freq=1))]:
    t=time.time(); s=cross_val_score(est,Xh,y,cv=3,scoring="accuracy")
    print(f"{name:24s} {s.mean():.4f}  {time.time()-t:.0f}s", flush=True)
