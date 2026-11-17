import numpy as np, time
from lightgbm import LGBMClassifier
from feats import *
d=np.load("emnist.npy")
te = slice(230000, 240000)
Xte = build2(d[te,1:]); yte = d[te,0]
Xall = build2(d[:120000,1:]); yall = d[:120000,0]
print("built", flush=True)
for N in [20000, 50000, 120000]:
    m = LGBMClassifier(n_estimators=600, learning_rate=0.1, num_leaves=63, subsample=0.8, subsample_freq=1,
                       max_bin=63, colsample_bytree=0.3, n_jobs=2, random_state=42, verbose=-1)
    t=time.time(); m.fit(Xall[:N], yall[:N])
    acc = (m.predict(Xte) == yte).mean()
    print(f"N={N:6d} holdout acc {acc:.4f}  {time.time()-t:.0f}s", flush=True)
