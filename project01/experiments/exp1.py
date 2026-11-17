import numpy as np, time
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from feats import *
d=np.load("emnist.npy"); N=20000
X, y = d[:N,1:], d[:N,0]
Xr=X.astype(np.float32)/255; Xs=build(X,False); Xh=build(X,True)
lg=lambda: LGBMClassifier(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3, n_jobs=2, random_state=42, verbose=-1)
for name, est, XX in [("LGBM raw",lg(),Xr),("LGBM simple",lg(),Xs),("LGBM simple+HOG",lg(),Xh),
                      ("SVC raw",SVC(),Xr),("SVC simple+HOG",SVC(),Xh/Xh.std(0).clip(1e-6))]:
    t=time.time(); s=cross_val_score(est,XX,y,cv=3,scoring="accuracy")
    print(f"{name:20s} {s.mean():.4f}  {time.time()-t:.0f}s", flush=True)
