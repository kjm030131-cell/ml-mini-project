# 16차: 15차 최고 SVC(오답 88)를 Stacking 재료로 → 기존 LGBM·원본SVC·옛SVC OOF와 합치기
import numpy as np, time, itertools
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
import exp15 as E  # base(), profiles() 재사용 (exp15 는 import 시 실행되지 않도록 아래에서 처리)
y=np.load("st_y.npy"); yte=np.load("st_yte.npy")
assert (y==np.load("fcache/tr_y.npy")).all() and (yte==np.load("fcache/te_y.npy")).all()
A=np.hstack([E.base("tr"),E.profiles("tr")]).astype(np.float64); B=np.hstack([E.base("te"),E.profiles("te")]).astype(np.float64)
g=1.2/(A.shape[1]*A.var())
t=time.time()
oof=cross_val_predict(SVC(C=10,gamma=g,cache_size=500),A,y,cv=3,method="decision_function",n_jobs=1)
m=SVC(C=10,gamma=g,cache_size=500).fit(A,y); te=m.decision_function(B)
np.save("st_oof_svc15.npy",oof); np.save("st_te_svc15.npy",te)
print(f"RESULT svc15 단독 wrong {(te.argmax(1)!=yte).sum()} {time.time()-t:.0f}s", flush=True)
O={k:np.load(f"st_oof_{k}.npy") for k in ["svc15","svc","lgbm","raw","lr"]}; T={k:np.load(f"st_te_{k}.npy") for k in O}
for c in [("svc15","lgbm"),("svc15","raw"),("svc15","lgbm","raw"),("svc15","svc","lgbm","raw"),("svc15","lgbm","raw","lr")]:
    for C in (1,10):
        w=(LogisticRegression(C=C,max_iter=3000).fit(np.hstack([O[k] for k in c]),y).predict(np.hstack([T[k] for k in c]))!=yte).sum()
        print(f"RESULT stack {'+'.join(c)} C={C} wrong {w}", flush=True)
