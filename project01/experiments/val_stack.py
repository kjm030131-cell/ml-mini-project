import json, sys, time, numpy as np
nb=json.load(open("/mnt/user-data/outputs/mini_proj_01_20221329_stack.ipynb"))
src=["".join(c["source"]) for c in nb["cells"]]
d=np.load("emnist.npy", mmap_mode="r")
n=int(sys.argv[1])
X_train=np.asarray(d[:, 1:]).astype(np.float32); y_train=np.asarray(d[:,0]).copy()
g=globals()
exec(src[3],g); exec(src[4],g); exec(src[5],g)
tr=slice(0,n); te=slice(220000,220000+(20000 if n>=50000 else 2000))
import resource
t=time.time(); stack_clf.fit(X_train[tr],y_train[tr]); p=stack_clf.predict(X_train[te])
print(f"RESULT notebook-stack n={n} wrong {(p!=y_train[te]).sum()} / {len(p)}  {time.time()-t:.0f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)
