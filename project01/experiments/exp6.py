import numpy as np, time
from sklearn.svm import SVC
from feats import *
d=np.load("emnist.npy"); tr=slice(0,50000); te=slice(230000,240000)
def parts(Xraw):
    X = Xraw.astype(np.float32)/255.0
    dk = deskew(np.ascontiguousarray(X.reshape(-1,28,28).transpose(0,2,1)))
    return dict(dkpix=dk.reshape(len(dk),-1), dhog4=hog_features(dk,4), dhog7=hog_features(dk,7),
                dh42=hog2(dk,4,2), dh84=hog2(dk,8,4))
P=parts(d[tr,1:]); Q=parts(d[te,1:]); ytr=d[tr,0]; yte=d[te,0]
def run(name, keys, C=10, gamma="scale"):
    A=np.hstack([P[k] for k in keys]); B=np.hstack([Q[k] for k in keys])
    t=time.time(); m=SVC(C=C, gamma=gamma).fit(A,ytr); t1=time.time()-t
    acc=(m.predict(B)==yte).mean()
    print(f"SVC C={C:<3} g={gamma} {name:26s} dim {A.shape[1]:5d} {acc:.4f} fit {t1:.0f}s total {time.time()-t:.0f}s", flush=True)
    return A.shape[1]*A.var()
run("dkpix+dhog4+dhog7", ["dkpix","dhog4","dhog7"])
run("dense HOG(4,2)", ["dh42"])
s=run("dkpix+dense HOG(4,2)", ["dkpix","dh42"])
run("dkpix+HOG(4,2)+HOG(8,4)", ["dkpix","dh42","dh84"])
run("dkpix+dense HOG(4,2)", ["dkpix","dh42"], C=30)
run("dkpix+dense HOG(4,2)", ["dkpix","dh42"], C=10, gamma=round(2.0/s,5))
