import numpy as np, time, sys
from sklearn.svm import SVC
y=np.load("fcache/tr_y.npy"); yte=np.load("fcache/te_y.npy")
def load(keys, part): return np.hstack([np.load(f"fcache/{part}_{k}.npy") for k in keys]).astype(np.float64)
def run(name, keys, tf=None, **kw):
    A=load(keys,"tr"); B=load(keys,"te")
    if tf: A, B = tf(A), tf(B)
    if kw.get("gamma")=="x2": kw["gamma"]=2/(A.shape[1]*A.var())
    t=time.time(); m=SVC(**({"C":10, "cache_size":500}|kw)).fit(A,y); acc=(m.predict(B)==yte).mean()
    print(f"RESULT {name:30s} {acc:.4f}  wrong {int((1-acc)*len(yte)):3d}/20000  {time.time()-t:.0f}s", flush=True)
sqrt = lambda Z: np.sqrt(np.clip(Z, 0, None))
base=["dkpix","h42","h84"]
if sys.argv[1]=="a":
    run("base (dkpix+h42+h84)", base)
    run("sqrt 변환", base, tf=sqrt)
    run("HOG 12방향", ["dkpix","h42b12","h84"])
    run("poly 3차", base, kernel="poly", degree=3, coef0=1)
elif sys.argv[1]=="b":
    run("크기정규화만 (snpix+sn42)", ["snpix","sn42"])
    run("gamma x2", base, gamma="x2")
    run("poly 4차", base, kernel="poly", degree=4, coef0=1)
else:
    run("base+크기정규화", base+["snpix","sn42"])
