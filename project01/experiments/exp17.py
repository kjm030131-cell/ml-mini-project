# 17차: Stacking 재료 다양화 — 같은 분할(학습 5만 3-fold OOF / 채점 2만)
# 사용법: python exp17.py svcG | et | rf | lgbm2 | meta
import sys, time, numpy as np, itertools
from sklearn.model_selection import cross_val_predict
from feats import hog2
import exp15 as E
y=np.load("st_y.npy"); yte=np.load("st_yte.npy")
def hog3(im, cell, stride, bins=9, signed=False, sigma=0.0):
    from scipy import ndimage
    if sigma>0: im=ndimage.gaussian_filter(im,(0,sigma,sigma))
    gy,gx=np.gradient(im,axis=(1,2)); mag=np.sqrt(gx**2+gy**2); rng=2*np.pi if signed else np.pi
    ang=(np.arctan2(gy,gx)%rng)/rng*bins-0.5; lo=np.floor(ang).astype(np.int32); w=(ang-lo).astype(np.float32)
    n=len(im); pos=range(0,28-cell+1,stride); h=np.zeros((n,len(pos),len(pos),bins),np.float32)
    for k in range(bins):
        m=mag*((lo%bins==k)*(1-w)+((lo+1)%bins==k)*w)
        c=np.cumsum(np.cumsum(np.pad(m,((0,0),(1,0),(1,0))),axis=1),axis=2)
        for a,i in enumerate(pos):
            for b,j in enumerate(pos): h[:,a,b,k]=c[:,i+cell,j+cell]-c[:,i,j+cell]-c[:,i+cell,j]+c[:,i,j]
    return (h/(np.sqrt((h**2).sum(-1,keepdims=True))+1e-6)).reshape(n,-1)
def F15(p): return np.hstack([E.base(p),E.profiles(p)]).astype(np.float32)
def FG(p):  # 실험 G 의 '흐림 + 부호 방향' HOG(오답 90)로 바꾼 SVC 재료
    im=E.img(p); return np.hstack([E.L(p,"dkpix"), hog3(im,4,2,18,True,0.7), hog3(im,8,4,18,True,0.7)*2, E.extra14(p)*1.5, E.profiles(p)]).astype(np.float32)
def save(name, o, t, sec):
    np.save(f"st_oof_{name}.npy",o); np.save(f"st_te_{name}.npy",t)
    print(f"RESULT {name} 단독 wrong {(t.argmax(1)!=yte).sum()} {sec:.0f}s", flush=True)
mode=sys.argv[1]; t0=time.time()
if mode=="svcG":
    from sklearn.svm import SVC
    A=FG("tr").astype(np.float64); B=FG("te").astype(np.float64); g=1.2/(A.shape[1]*A.var())
    o=cross_val_predict(SVC(C=10,gamma=g,cache_size=500),A,y,cv=3,method="decision_function")
    t=SVC(C=10,gamma=g,cache_size=500).fit(A,y).decision_function(B); save("svcG",o,t,time.time()-t0)
elif mode in ("et","rf"):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    M=ExtraTreesClassifier if mode=="et" else RandomForestClassifier
    A=F15("tr"); B=F15("te"); mk=lambda: M(n_estimators=400, max_features="sqrt", n_jobs=1, random_state=42)
    o=cross_val_predict(mk(),A,y,cv=3,method="predict_proba"); t=mk().fit(A,y).predict_proba(B); save(mode,o,t,time.time()-t0)
elif mode=="lgbm2":
    from lightgbm import LGBMClassifier
    A=F15("tr"); B=F15("te")
    mk=lambda: LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, max_bin=63, colsample_bytree=0.2, subsample=0.8, subsample_freq=1, n_jobs=1, random_state=7, verbose=-1)
    o=cross_val_predict(mk(),A,y,cv=3,method="predict_proba"); t=mk().fit(A,y).predict_proba(B); save("lgbm2",o,t,time.time()-t0)
else:
    from sklearn.linear_model import LogisticRegression
    import os
    names=[n for n in ["svc15","lgbm","raw","lr","svcG","et","rf","lgbm2","svc"] if os.path.exists(f"st_te_{n}.npy")]
    O={n:np.load(f"st_oof_{n}.npy") for n in names}; T={n:np.load(f"st_te_{n}.npy") for n in names}
    best=[]
    core=["svc15","lgbm","raw","lr"]
    extra=[n for n in names if n not in core]
    for r in range(0,len(extra)+1):
        for add in itertools.combinations(extra,r):
            c=tuple(core)+add
            for C in (0.3,1):
                w=(LogisticRegression(C=C,max_iter=4000).fit(np.hstack([O[k] for k in c]),y).predict(np.hstack([T[k] for k in c]))!=yte).sum()
                best.append((w,'+'.join(c),C))
    for w,c,C in sorted(best)[:12]: print(f"RESULT stack {c} C={C} wrong {w}", flush=True)
    allw=np.all([T[n].argmax(1)!=yte for n in names],axis=0).sum(); print(f"RESULT 모든 모델이 틀린 문제 {allw}", flush=True)
