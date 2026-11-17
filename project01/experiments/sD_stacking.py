# 세션 D: 사이킷런 StackingClassifier로 Stacking 재현 + 학습 시간 측정
import sys, time, numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from common import feat, labels
LOG = "../results/D_stacking.md"
y, yte = labels()
keys = [("dkpix",1),("h42",1),("h84",2),("rawpix",1),("rh42",1),("holes",1)]
A = np.hstack([feat("tr",k)*w for k,w in keys]).astype(np.float64); B = np.hstack([feat("te",k)*w for k,w in keys]).astype(np.float64)
n_dk = 784+1521+324; n_raw = 784+1521
sel = lambda a,b: ColumnTransformer([("s","passthrough",list(range(a,b)))])
svc_dk = make_pipeline(sel(0,n_dk), SVC(C=10, cache_size=500))
svc_raw = make_pipeline(sel(n_dk,n_dk+n_raw), SVC(C=10, cache_size=500))
lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3, n_jobs=2, random_state=42, verbose=-1)
mode = sys.argv[1]
if mode == "0":
    est = [("svc", svc_dk), ("lgbm", lgbm), ("svc_raw", svc_raw)]
    clf = StackingClassifier(est, final_estimator=LogisticRegression(C=10, max_iter=3000), cv=3, n_jobs=1)
    t=time.time(); clf.fit(A,y); w=int((clf.predict(B)!=yte).sum())
    line=f"| StackingClassifier(SVC+LGBM+원본SVC, cv=3) | {w} | {1-w/len(yte):.4f} | {time.time()-t:.0f}s |"
else:
    clf = StackingClassifier([("svc", svc_dk), ("lgbm", lgbm)], final_estimator=LogisticRegression(C=10, max_iter=3000), cv=3, n_jobs=1)
    t=time.time(); clf.fit(A,y); w=int((clf.predict(B)!=yte).sum())
    line=f"| StackingClassifier(SVC+LGBM, cv=3) | {w} | {1-w/len(yte):.4f} | {time.time()-t:.0f}s |"
print(line, flush=True); open(LOG,"a").write(line+"\n")
