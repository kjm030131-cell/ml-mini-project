import numpy as np
from scipy import ndimage
def sizenorm(imgs, box=20):
    out=np.zeros_like(imgs)
    for i,a in enumerate(imgs):
        r=np.where(a.max(1)>0.1)[0]; c=np.where(a.max(0)>0.1)[0]
        if len(r)==0: continue
        crop=a[r[0]:r[-1]+1, c[0]:c[-1]+1]; h,w=crop.shape; s=box/max(h,w)
        z=ndimage.zoom(crop, s, order=1).clip(0,1); h2,w2=z.shape
        canvas=np.zeros((28,28),np.float32); oy=(28-h2)//2; ox=(28-w2)//2; canvas[oy:oy+h2, ox:ox+w2]=z[:28-oy,:28-ox]
        cy,cx=ndimage.center_of_mass(canvas); out[i]=ndimage.shift(canvas,(14-cy,14-cx),order=1)
    return out

import numpy as np, time
exec(open('/home/claude/exp1/feat.py').read())
from sklearn.svm import SVC
d=np.load('/home/claude/work/ML1/d.npy'); idx=np.random.RandomState(7).permutation(len(d))[100000:200000]
X=d[idx,1:].astype(np.float32); y=d[idx,0]; del d
dk=deskew(orient(X)); P=lambda a:a.reshape(len(a),-1)
F=np.hstack([P(dk),stroke_features(dk)*1.5,profiles(dk),P(sizenorm(dk))]).astype(np.float64)
g=2.0/(F.shape[1]*F[:60000].var()); t=time.time()
p=SVC(C=10,gamma=g,cache_size=800).fit(F[:60000],y[:60000]).predict(F[60000:]); np.save('vpred_V_A5.npy',p)
line=f"| V_A5_base+sizenorm (구간2) | {F.shape[1]} | C=10 g={g:.4g} | {(p!=y[60000:]).sum()} | {(p==y[60000:]).mean():.5f} | {time.time()-t:.0f}s |"; print(line,flush=True); open('results.md','a').write(line+'\n')
