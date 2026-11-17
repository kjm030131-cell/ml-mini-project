import numpy as np
from feats import *
d=np.load("emnist.npy", mmap_mode="r")
for name, sl in [("tr", slice(0,50000)), ("te", slice(220000,240000))]:
    P=parts_all(np.asarray(d[sl,1:]))
    for k,v in P.items(): np.save(f"fcache/{name}_{k}.npy", v)
    np.save(f"fcache/{name}_y.npy", np.asarray(d[sl,0]))
    del P
print("prep done", flush=True)
