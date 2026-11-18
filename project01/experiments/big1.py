# 큰 학습(16만)에서 실제로 효과 있는 설정 찾기: 학습 0~160000, 채점 220000~240000
import json, sys, time, numpy as np
nb=json.load(open("/mnt/user-data/outputs/mini_proj_01_20221329_stack.ipynb"))
src=["".join(c["source"]) for c in nb["cells"]]
g=globals(); exec(src[3],g)
exec(src[4].split("F = feat_svc15")[0],g)
d=np.load("emnist.npy", mmap_mode="r"); N=int(sys.argv[1])
Xtr=np.asarray(d[:N,1:]).astype(np.float32); ytr=np.asarray(d[:N,0]).copy()
Xte=np.asarray(d[220000:240000,1:]).astype(np.float32); yte=np.asarray(d[220000:240000,0]).copy()
from sklearn.svm import SVC
def run(name, fe, C=10, gm=None):
    t=time.time(); A=fe(Xtr); B=fe(Xte); m1=sum(A[i:i+10000].sum() for i in range(0,len(A),10000))/A.size; m2=sum((A[i:i+10000]**2).sum() for i in range(0,len(A),10000))/A.size; g0=1/(A.shape[1]*(m2-m1*m1)); gam=(gm or 1.0)*g0
    m=SVC(C=C,gamma=gam,cache_size=500).fit(A,ytr); w=(m.predict(B)!=yte).sum()
    print(f"RESULT N={N} {name:28s} wrong {w} / 20000  {time.time()-t:.0f}s", flush=True); del A,B,m
for name,fe,C,gm in [("기준(보정+HOG2) gamma1",feat_lr,10,1.0),("svc15 gamma1.2",feat_svc15,10,1.2),
                      ("svc15 gamma2",feat_svc15,10,2.0),("svc15 gamma1.2 C30",feat_svc15,30,1.2),("기준 gamma2",feat_lr,10,2.0)]:
    run(name,fe,C,gm)
