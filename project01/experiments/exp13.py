# 강의자료 기반 추가 실험: StandardScaler(W11), 구멍 개수 특성(W06 31쪽)
import numpy as np, time, sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from feats import holes
y=np.load("fcache/tr_y.npy"); yte=np.load("fcache/te_y.npy")
L=lambda p,k: np.load(f"fcache/{p}_{k}.npy")
A0=np.hstack([L("tr",k) for k in ["dkpix","h42","h84"]]); B0=np.hstack([L("te",k) for k in ["dkpix","h42","h84"]])
def run(name, A, B, **kw):
    A=A.astype(np.float64); B=B.astype(np.float64)
    t=time.time(); m=SVC(**({"C":10,"cache_size":500}|kw)).fit(A,y); w=(m.predict(B)!=yte).sum()
    print(f"RESULT {name:36s} wrong {w}  {time.time()-t:.0f}s", flush=True)
if sys.argv[1]=="scale":
    sc=StandardScaler().fit(A0); run("StandardScaler (W11)", sc.transform(A0), sc.transform(B0))
else:
    d=np.load("emnist.npy", mmap_mode="r")
    Htr=holes(np.asarray(d[0:50000,1:]).astype(np.float32).reshape(-1,28,28)/255)
    Hte=holes(np.asarray(d[220000:240000,1:]).astype(np.float32).reshape(-1,28,28)/255)
    Htr[:,1]/=50; Hte[:,1]/=50
    for w in [0.5, 1.5]:
        run(f"+holes x{w} (W06 p31)", np.hstack([A0, Htr*w]), np.hstack([B0, Hte*w]))
