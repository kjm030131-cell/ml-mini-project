import numpy as np, time, os, sys
from scipy import ndimage
exec(open('/home/claude/exp1/feat.py').read())   # deskew, holes, simple, stroke_features, profiles (HOG 제외), orient
def load():
    if os.path.exists('sub.npz'):
        z=np.load('sub.npz'); return z['im'],z['dk'],z['y']
    d=np.load('/home/claude/work/ML1/d.npy'); idx=np.random.RandomState(7).permutation(len(d))[:100000]
    X=d[idx,1:].astype(np.float32); y=d[idx,0]
    im=orient(X); dk=deskew(im); np.savez('sub.npz',im=im,dk=dk,y=y); return im,dk,y
im,dk,y=load(); NTR=60000
from sklearn.svm import SVC
def run(name,F,C=10,gm=2.0,gamma=None):
    F=F.astype(np.float64); g=gamma or gm/(F.shape[1]*F[:NTR].var())
    t=time.time(); m=SVC(C=C,gamma=g,cache_size=800).fit(F[:NTR],y[:NTR]); p=m.predict(F[NTR:])
    err=int((p!=y[NTR:]).sum()); 
    line=f"| {name} | {F.shape[1]} | C={C} g={g:.4g} | {err} | {1-err/len(p):.5f} | {time.time()-t:.0f}s |"
    print(line,flush=True); open('results.md','a').write(line+'\n'); np.save(f'pred_{name}.npy',p); return err
