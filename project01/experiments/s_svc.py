# 스태킹 재료 1: SVC, LR (특성 F = 보정 픽셀 + 촘촘한 HOG + 큰 칸 HOG). 학습 5만 / 홀드아웃 2만
import numpy as np, time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from feats import *
d=np.load("emnist.npy"); tr=slice(0,50000); te=slice(220000,240000)
def f(Xraw, chunk=10000):
    out=[]
    for s in range(0,len(Xraw),chunk):
        X = Xraw[s:s+chunk].astype(np.float32)/255.0
        dk = deskew(np.ascontiguousarray(X.reshape(-1,28,28).transpose(0,2,1)))
        out.append(np.hstack([dk.reshape(len(dk),-1), hog2(dk,4,2), hog2(dk,8,4)]).astype(np.float32))
    return np.vstack(out)
A=f(d[tr,1:]); B=f(d[te,1:]); y=d[tr,0].copy(); yte=d[te,0].copy(); del d
np.save("st_y.npy",y); np.save("st_yte.npy",yte)
t=time.time(); m=LogisticRegression(C=10, max_iter=300)
np.save("st_oof_lr.npy", cross_val_predict(m,A,y,cv=3,method="predict_proba")); m.fit(A,y); p=m.predict_proba(B); np.save("st_te_lr.npy",p)
print(f"LR  holdout {(p.argmax(1)==yte).mean():.4f} {time.time()-t:.0f}s", flush=True)
t=time.time(); m=SVC(C=10)
np.save("st_oof_svc.npy", cross_val_predict(m,A,y,cv=3,method="decision_function")); m.fit(A,y); p=m.decision_function(B); np.save("st_te_svc.npy",p)
print(f"SVC holdout {(m.predict(B)==yte).mean():.4f} {time.time()-t:.0f}s", flush=True)
