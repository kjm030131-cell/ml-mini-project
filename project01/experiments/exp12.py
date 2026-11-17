# 하이퍼파라미터 탐색: 특성 묶음 비중, C, gamma (학습 5만 / 홀드아웃 2만)
import numpy as np, time, sys
from sklearn.svm import SVC
y=np.load("fcache/tr_y.npy"); yte=np.load("fcache/te_y.npy")
L=lambda part,k: np.load(f"fcache/{part}_{k}.npy")
TR={k:L("tr",k) for k in ["dkpix","h42","h84"]}; TE={k:L("te",k) for k in ["dkpix","h42","h84"]}
n=sum(v.size for v in TR.values()); mu=sum(v.sum(dtype=np.float64) for v in TR.values())/n
var=sum(((v.astype(np.float64)-mu)**2).sum() for v in TR.values())/n
base_g = 1.0/(2629*var)
def run(wp, w42, w84, C=10, gm=1.0):
    A=np.hstack([TR["dkpix"]*wp, TR["h42"]*w42, TR["h84"]*w84]).astype(np.float64); B=np.hstack([TE["dkpix"]*wp, TE["h42"]*w42, TE["h84"]*w84]).astype(np.float64)
    t=time.time(); m=SVC(C=C, gamma=base_g*gm, cache_size=500).fit(A,y); w=(m.predict(B)!=yte).sum()
    print(f"RESULT pix={wp} h42={w42} h84={w84} C={C} gamma={gm}x  wrong {w}  {time.time()-t:.0f}s", flush=True)
grid = {
 "a": [(1,1,1,10,1.0),(0.7,1,1,10,1.0),(1.4,1,1,10,1.0),(1,1,1.5,10,1.0),(1,1,1,5,1.0),(1,1,1,10,1.5)],
 "b": [(1,1,2,10,1.0),(0.7,1,1.5,10,1.0),(1,1,0.6,10,1.0),(1,1,1,20,1.0),(1,1,1,10,0.7),(0.5,1,1,10,1.0)],
}[sys.argv[1]]
for g in grid: run(*g)
