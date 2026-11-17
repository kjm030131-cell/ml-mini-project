# 데이터 양에 따른 SVC 점수/시간: 촘촘한 HOG(보정 이미지)
import numpy as np, time
from sklearn.svm import SVC
from feats import *
d=np.load("emnist.npy"); te=slice(230000,240000)
def f(Xraw, chunk=10000):
    out=[]
    for s in range(0,len(Xraw),chunk):
        X = Xraw[s:s+chunk].astype(np.float32)/255.0
        dk = deskew(np.ascontiguousarray(X.reshape(-1,28,28).transpose(0,2,1)))
        out.append(hog2(dk,4,2).astype(np.float64))
    return np.ascontiguousarray(np.vstack(out))
t=time.time(); A=f(d[:120000,1:]); B=f(d[te,1:]); y=d[:120000,0]; yte=d[te,0]
del d
print(f"feat {time.time()-t:.0f}s", flush=True)
t=time.time(); m=SVC(C=10).fit(A,y); t1=time.time()-t
acc=(m.predict(B)==yte).mean()
print(f"SVC denseHOG N=120000 holdout {acc:.4f} fit {t1:.0f}s total {time.time()-t:.0f}s nSV {m.n_support_.sum()}", flush=True)
