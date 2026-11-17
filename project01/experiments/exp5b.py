# LGBM 계열(1코어): 학습 5만 / 고정 홀드아웃 1만
import numpy as np, time
from lightgbm import LGBMClassifier
from feats import *
d=np.load("emnist.npy"); tr=slice(0,50000); te=slice(230000,240000)
ytr=d[tr,0]; yte=d[te,0]
for name, fn in [("build3 (+deskew)", build3), ("build2", build2)]:
    A=fn(d[tr,1:]); B=fn(d[te,1:])
    m=LGBMClassifier(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3, n_jobs=1, random_state=42, verbose=-1)
    t=time.time(); m.fit(A,ytr); acc=(m.predict(B)==yte).mean()
    print(f"LGBM300 {name:20s} {acc:.4f} {time.time()-t:.0f}s", flush=True)
    np.save(f"proba_{fn.__name__}.npy", m.predict_proba(B))
