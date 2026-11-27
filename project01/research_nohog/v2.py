import numpy as np, time
exec(open('/home/claude/exp1/feat.py').read())
from sklearn.svm import SVC
d=np.load('/home/claude/work/ML1/d.npy'); idx=np.random.RandomState(7).permutation(len(d))[100000:200000]
X=d[idx,1:].astype(np.float32); y=d[idx,0]; del d
dk=deskew(orient(X)); S=stroke_features(dk)*1.5; Pr=profiles(dk); P=lambda a:a.reshape(len(a),-1)
for name,F in [('V_A0_base',np.hstack([P(dk),S,Pr])),('V_B4_sqrt',np.hstack([P(np.sqrt(dk.clip(0))),S,Pr]))]:
    F=F.astype(np.float64); g=2.0/(F.shape[1]*F[:60000].var()); t=time.time()
    p=SVC(C=10,gamma=g,cache_size=800).fit(F[:60000],y[:60000]).predict(F[60000:]); np.save(f'vpred_{name}.npy',p); np.save('vy.npy',y[60000:])
    line=f"| {name} (구간2) | {F.shape[1]} | C=10 g={g:.4g} | {(p!=y[60000:]).sum()} | {(p==y[60000:]).mean():.5f} | {time.time()-t:.0f}s |"; print(line,flush=True); open('results.md','a').write(line+'\n')
