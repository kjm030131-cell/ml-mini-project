# Stacking 재료 추가: 서로 다른 시각의 SVC (학습 5만 3-fold oof / 홀드아웃 2만)
import numpy as np, time, sys
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from feats import *
y=np.load("st_y.npy"); yte=np.load("st_yte.npy")
which=sys.argv[1]
if which=="sn":
    keys=["snpix","sn42"]
    A=np.hstack([np.load(f"fcache/tr_{k}.npy") for k in keys]).astype(np.float64)
    B=np.hstack([np.load(f"fcache/te_{k}.npy") for k in keys]).astype(np.float64)
else:
    d=np.load("emnist.npy", mmap_mode="r")
    def f(Xraw):
        out=[]
        for s in range(0,len(Xraw),10000):
            X=np.asarray(Xraw[s:s+10000]).astype(np.float32)/255.0
            im=np.ascontiguousarray(X.reshape(-1,28,28).transpose(0,2,1))
            out.append(np.hstack([im.reshape(len(im),-1), hog2(im,4,2), hog2(im,8,4)]))
        return np.vstack(out).astype(np.float64)
    A=f(d[0:50000,1:]); B=f(d[220000:240000,1:])
t=time.time(); m=SVC(C=10, cache_size=500)
np.save(f"st_oof_{which}.npy", cross_val_predict(m,A,y,cv=3,method="decision_function"))
m.fit(A,y); p=m.decision_function(B); np.save(f"st_te_{which}.npy",p)
print(f"RESULT svc_{which} holdout {(p.argmax(1)==yte).mean():.4f} wrong {(p.argmax(1)!=yte).sum()} {time.time()-t:.0f}s", flush=True)
