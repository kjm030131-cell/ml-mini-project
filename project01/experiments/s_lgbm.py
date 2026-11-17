# 스태킹 재료 2: LGBM (build3 특성)
import numpy as np, time
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict
from feats import *
d=np.load("emnist.npy"); tr=slice(0,50000); te=slice(220000,240000)
def f(Xraw, chunk=10000): return np.vstack([build3(Xraw[s:s+chunk]) for s in range(0,len(Xraw),chunk)])
A=f(d[tr,1:]); B=f(d[te,1:]); y=d[tr,0].copy(); yte=d[te,0].copy(); del d
t=time.time(); m=LGBMClassifier(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3, n_jobs=2, random_state=42, verbose=-1)
np.save("st_oof_lgbm.npy", cross_val_predict(m,A,y,cv=3,method="predict_proba")); m.fit(A,y); p=m.predict_proba(B); np.save("st_te_lgbm.npy",p)
print(f"LGBM holdout {(p.argmax(1)==yte).mean():.4f} {time.time()-t:.0f}s", flush=True)
