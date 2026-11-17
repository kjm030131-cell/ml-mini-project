# 15차: 3↔5 겨냥 윤곽선(프로파일) 특성 + 지금까지 좋았던 것 전부 합치기
import numpy as np, time, sys
from sklearn.svm import SVC
from collections import Counter
from feats import holes, hog2
y=np.load("fcache/tr_y.npy"); yte=np.load("fcache/te_y.npy")
L=lambda p,k: np.load(f"fcache/{p}_{k}.npy")
def img(p): return L(p,"dkpix").reshape(-1,28,28)
def extra14(p):
    im=img(p); b=(im>0.3).astype(np.int8)
    tr_r=(np.diff(b,axis=2)==1).sum(2)/4.0; tr_c=(np.diff(b,axis=1)==1).sum(1)/4.0
    top=im[:,:10,:].sum((1,2))/im.sum((1,2)).clip(1e-6); left=im[:,:,:14].sum((1,2))/im.sum((1,2)).clip(1e-6)
    h=holes(im)/np.array([1,50],np.float32)
    return np.hstack([tr_r,tr_c,top[:,None],left[:,None],h]).astype(np.float32)
def profiles(p):
    # 행마다 첫/마지막 획 위치(왼쪽·오른쪽 윤곽), 열마다 첫/마지막 획 위치(위·아래 윤곽) → 112개
    b=img(p)>0.3; n=len(b); idx=np.arange(28)
    def first(m, axis):
        has=m.any(axis); f=np.where(has, np.argmax(m,axis), 28); return f/28.0
    left=first(b,2); right=1-first(b[:,:,::-1],2); top=first(b,1); bottom=1-first(b[:,::-1,:],1)
    return np.hstack([left,right,top,bottom]).astype(np.float32)
def h73(p): return hog2(img(p),7,3)
base=lambda p: np.hstack([L(p,"dkpix"), L(p,"h42"), L(p,"h84")*2, extra14(p)*1.5])
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=="prof": A=np.hstack([base("tr"),profiles("tr")]); B=np.hstack([base("te"),profiles("te")])
    else: A=np.hstack([base("tr"),h73("tr"),profiles("tr")]); B=np.hstack([base("te"),h73("te"),profiles("te")])
    A=A.astype(np.float64); B=B.astype(np.float64)
    t=time.time(); m=SVC(C=10, gamma=1.2/(A.shape[1]*A.var()), cache_size=500).fit(A,y); p=m.predict(B); w=(p!=yte).sum()
    print(f"RESULT {mode:5s} wrong {w}  {time.time()-t:.0f}s", flush=True)
    print(Counter(tuple(sorted((int(a),int(b)))) for a,b in zip(yte[p!=yte],p[p!=yte])).most_common(6), flush=True)
    