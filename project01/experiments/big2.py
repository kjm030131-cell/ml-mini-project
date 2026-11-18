# 학습 0~120000, 채점 120000~220000 (10만 개: 오차가 작아 차이를 더 정확히 봄)
import json, sys, time, numpy as np
nb=json.load(open("/mnt/user-data/outputs/mini_proj_01_20221329_stack.ipynb"))
src=["".join(c["source"]) for c in nb["cells"]]
g=globals(); exec(src[3],g); exec(src[4].split("F = feat_svc15")[0],g)
d=np.load("emnist.npy", mmap_mode="r")
from sklearn.svm import SVC
def one_svc15_h73(X):
    dk = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
    return np.hstack([dk.reshape(len(dk), -1), hog(dk, 4, 2), hog(dk, 8, 4) * 2, hog(dk, 7, 3),
                      stroke_features(dk) * 1.5, profiles(dk)])
def one_svc15_b12(X):
    dk = deskew(np.ascontiguousarray(X.reshape(-1, 28, 28).transpose(0, 2, 1)))
    return np.hstack([dk.reshape(len(dk), -1), hog(dk, 4, 2, 12), hog(dk, 8, 4, 12) * 2,
                      stroke_features(dk) * 1.5, profiles(dk)])
C={"svc15":one_svc15,"svc15+h73":one_svc15_h73,"svc15 hog12방향":one_svc15_b12}
ytr=np.asarray(d[:120000,0]).copy(); yte=np.asarray(d[120000:220000,0]).copy()
for name in sys.argv[1:]:
    t=time.time(); A=by_chunk(C[name],np.asarray(d[:120000,1:]).astype(np.float32))
    m1=sum(A[i:i+10000].sum() for i in range(0,len(A),10000))/A.size; m2=sum((A[i:i+10000]**2).sum() for i in range(0,len(A),10000))/A.size
    m=SVC(C=10,gamma=1.2/(A.shape[1]*(m2-m1*m1)),cache_size=500).fit(A,ytr); del A
    w=0
    for s in range(120000,220000,20000):
        B=by_chunk(C[name],np.asarray(d[s:s+20000,1:]).astype(np.float32)); w+=(m.predict(B)!=yte[s-120000:s-100000]).sum()
    print(f"RESULT {name:18s} wrong {w} / 100000  acc {1-w/1e5:.5f}  {time.time()-t:.0f}s", flush=True)
