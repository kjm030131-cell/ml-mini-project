# 세션 I: 좋아진 SVC 특성으로 수동 Stacking 검증 (StackingClassifier와 같은 원리, 메모리 절약형)
# 사용법: python sI_stack.py svc | lgbm | raw   (각각 3-fold OOF + 전체 학습 → 홀드아웃 점수 저장)
#         python sI_stack.py meta                 (저장된 결과로 LogisticRegression 최종 판정)
import sys, time, os, numpy as np
from common import feat, labels
OUT = "../results/I_cache"; os.makedirs(OUT, exist_ok=True); LOG = "../results/I_stack.md"
y, yte = labels()
def X(part, spec): return np.hstack([feat(part, k) * w for k, w in spec])
SVC_SPEC = [("dkpix",1),("h42",1),("h84",2),("holes",1),("h73",1)]
RAW_SPEC = [("rawpix",1),("rh42",1)]
def log(line): print(line, flush=True); open(LOG, "a").write(line + "\n")
def oof(name, make, spec, method):
    from sklearn.model_selection import cross_val_predict
    A = X("tr", spec); B = X("te", spec)
    if name != "lgbm": A, B = A.astype(np.float64), B.astype(np.float64)
    t = time.time(); m = make(A)
    o = cross_val_predict(m, A, y, cv=3, method=method); m.fit(A, y); p = getattr(m, method)(B)
    np.save(f"{OUT}/oof_{name}.npy", o); np.save(f"{OUT}/te_{name}.npy", p)
    log(f"| {name} 단독 | {int((p.argmax(1) != yte).sum())} | {(p.argmax(1) == yte).mean():.4f} | {time.time()-t:.0f}s |")
mode = sys.argv[1]
if mode in ("svc", "raw"):
    from sklearn.svm import SVC
    spec = SVC_SPEC if mode == "svc" else RAW_SPEC
    oof(mode, lambda A: SVC(C=10, gamma=1.2 / (A.shape[1] * A.var()), cache_size=500), spec, "decision_function")
elif mode == "lgbm":
    from lightgbm import LGBMClassifier
    oof("lgbm", lambda A: LGBMClassifier(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3,
                                          n_jobs=2, random_state=42, verbose=-1), SVC_SPEC, "predict_proba")
else:
    from sklearn.linear_model import LogisticRegression
    import itertools
    names = [n for n in ["svc", "lgbm", "raw"] if os.path.exists(f"{OUT}/te_{n}.npy")]
    O = {n: np.load(f"{OUT}/oof_{n}.npy") for n in names}; T = {n: np.load(f"{OUT}/te_{n}.npy") for n in names}
    for r in range(2, len(names) + 1):
        for c in itertools.combinations(names, r):
            for C in (1, 10):
                m = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([O[k] for k in c]), y)
                w = int((m.predict(np.hstack([T[k] for k in c])) != yte).sum())
                log(f"| Stacking {'+'.join(c)} (LR C={C}) | {w} | {1 - w / len(yte):.4f} | - |")
