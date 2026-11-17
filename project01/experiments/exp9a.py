# 다른 모델들: 보정 픽셀 + 촘촘한 HOG + 큰 칸 HOG, 학습 5만 / 홀드아웃 1만
import numpy as np, time
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from feats import *
d=np.load("emnist.npy"); tr=slice(0,50000); te=slice(230000,240000)
def f(Xraw):
    X = Xraw.astype(np.float32)/255.0
    dk = deskew(np.ascontiguousarray(X.reshape(-1,28,28).transpose(0,2,1)))
    return np.hstack([dk.reshape(len(dk),-1), hog2(dk,4,2), hog2(dk,8,4)]).astype(np.float32)
A=f(d[tr,1:]); B=f(d[te,1:]); ytr=d[tr,0]; yte=d[te,0]; del d
np.save("F_tr.npy",A); np.save("F_te.npy",B)
for name, m in [("ExtraTrees 300", ExtraTreesClassifier(n_estimators=300, n_jobs=1, random_state=42)),
                ("RandomForest 300", RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=42)),
                ("LogisticRegression C=10", LogisticRegression(C=10, max_iter=300))]:
    t=time.time(); m.fit(A,ytr); p=m.predict_proba(B)
    print(f"{name:26s} {(p.argmax(1)==yte).mean():.4f}  {time.time()-t:.0f}s", flush=True)
    np.save("proba_"+name.split()[0]+".npy", p)
