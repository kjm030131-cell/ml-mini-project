# 오답 숫자쌍(3↔5, 5↔8, 1↔7, 7↔9) 겨냥 특성: 행·열 획 전환 횟수 + 구멍 + 위쪽 영역 잉크
import numpy as np, time, sys
from sklearn.svm import SVC
from feats import holes
y=np.load("fcache/tr_y.npy"); yte=np.load("fcache/te_y.npy")
L=lambda p,k: np.load(f"fcache/{p}_{k}.npy")
def base(p): return np.hstack([L(p,"dkpix"), L(p,"h42"), L(p,"h84")*2])
def extra(p):
    im=L(p,"dkpix").reshape(-1,28,28); b=(im>0.3).astype(np.int8)
    tr_r=(np.diff(b,axis=2)==1).sum(2)/4.0; tr_c=(np.diff(b,axis=1)==1).sum(1)/4.0   # 행/열별 0→1 전환 횟수 (28+28)
    top=im[:,:10,:].sum((1,2))/im.sum((1,2)).clip(1e-6); left=im[:,:,:14].sum((1,2))/im.sum((1,2)).clip(1e-6)
    h=holes(im)/np.array([1,50],np.float32)
    return np.hstack([tr_r,tr_c,top[:,None],left[:,None],h]).astype(np.float32)
mode=sys.argv[1]
A=base("tr"); B=base("te")
if mode in ("tr","all"): A=np.hstack([A, extra("tr")*(1 if mode=="tr" else 1.5)]); B=np.hstack([B, extra("te")*(1 if mode=="tr" else 1.5)])
A=A.astype(np.float64); B=B.astype(np.float64)
t=time.time(); m=SVC(C=10, gamma=1.0/(A.shape[1]*A.var()), cache_size=500).fit(A,y); p=m.predict(B); w=(p!=yte).sum()
print(f"RESULT {mode:5s} wrong {w}  {time.time()-t:.0f}s", flush=True)
from collections import Counter
print(Counter(tuple(sorted((int(a),int(b)))) for a,b in zip(yte[p!=yte],p[p!=yte])).most_common(5), flush=True)
