# LGBM + SVC 합치기(soft voting) 추정: 학습 5만 / 홀드아웃 1만
import numpy as np, time
from sklearn.svm import SVC
from feats import *
d=np.load("emnist.npy"); tr=slice(0,50000); te=slice(230000,240000)
def f(Xraw):
    X = Xraw.astype(np.float32)/255.0
    dk = deskew(np.ascontiguousarray(X.reshape(-1,28,28).transpose(0,2,1)))
    return np.hstack([dk.reshape(len(dk),-1), hog_features(dk,4)])
A=f(d[tr,1:]); B=f(d[te,1:]); ytr=d[tr,0]; yte=d[te,0]
t=time.time(); m=SVC(C=10, probability=True, random_state=42).fit(A,ytr)
ps=m.predict_proba(B); np.save("proba_svc.npy", ps)
pl=np.load("proba_build3.npy")
print(f"SVC proba alone      {(ps.argmax(1)==yte).mean():.4f}  {time.time()-t:.0f}s", flush=True)
print(f"LGBM alone           {(pl.argmax(1)==yte).mean():.4f}", flush=True)
for w in [0.3,0.5,0.7]:
    print(f"VOTE w_svc={w}       {((w*ps+(1-w)*pl).argmax(1)==yte).mean():.4f}", flush=True)
print("both wrong", ((ps.argmax(1)!=yte)&(pl.argmax(1)!=yte)).sum(), "svc wrong", (ps.argmax(1)!=yte).sum(), "lgbm wrong", (pl.argmax(1)!=yte).sum(), flush=True)
