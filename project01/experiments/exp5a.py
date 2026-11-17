# SVC 계열: 학습 5만 / 고정 홀드아웃 1만
import numpy as np, time
from sklearn.svm import SVC
from feats import *
d=np.load("emnist.npy"); tr=slice(0,50000); te=slice(230000,240000)
def parts(Xraw):
    X = Xraw.astype(np.float32)/255.0
    imgs = X.reshape(-1,28,28)
    dk = deskew(np.ascontiguousarray(imgs.transpose(0,2,1)))
    return dict(dkpix=dk.reshape(len(dk),-1), hog4=hog_features(imgs,4), hog7=hog_features(imgs,7),
                dhog4=hog_features(dk,4), dhog7=hog_features(dk,7), raw=X)
P=parts(d[tr,1:]); Q=parts(d[te,1:]); ytr=d[tr,0]; yte=d[te,0]
def run(name, keys, C=10):
    A=np.hstack([P[k] for k in keys]); B=np.hstack([Q[k] for k in keys])
    t=time.time(); m=SVC(C=C).fit(A,ytr); t1=time.time()-t
    acc=(m.predict(B)==yte).mean()
    print(f"SVC C={C:<3} {name:28s} {acc:.4f} fit {t1:.0f}s total {time.time()-t:.0f}s nSV {m.n_support_.sum()}", flush=True)
run("deskew pixels", ["dkpix"])
run("deskew HOG4+7", ["dhog4","dhog7"])
run("HOG4 + deskew HOG4", ["hog4","dhog4"])
run("deskew pix + deskew HOG4", ["dkpix","dhog4"])
run("raw pixels", ["raw"])
