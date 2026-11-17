import numpy as np, time
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from feats import *
d=np.load("emnist.npy"); N=20000
X, y = d[:N,1:], d[:N,0]
X3=build3(X); print("features", X3.shape, flush=True)
m=LGBMClassifier(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3, n_jobs=2, random_state=42, verbose=-1)
t=time.time(); s=cross_val_score(m,X3,y,cv=3,scoring="accuracy")
print(f"HOG+holes+deskew 300     {s.mean():.4f}  {time.time()-t:.0f}s", flush=True)
