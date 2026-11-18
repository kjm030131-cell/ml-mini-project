import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
d=pd.read_csv('/home/claude/ml-mini-project/data/project02/train.csv.gz')
Xdf=d.drop(columns=['subject','Activity']); y=d['Activity'].values; g=d['subject']
roll=lambda df,w: df.groupby(g).transform(lambda s: s.rolling(w,center=True,min_periods=1).mean()).values
Zc=Xdf-Xdf.groupby(g).transform('mean'); Z=Zc.values
def ev(name,X,cv=10,m=None):
    m=m or SVC(C=100); s=cross_val_score(m,X,y,cv=cv,n_jobs=2); print(f"{name:34s} {s.mean():.4f} min {s.min():.3f}",flush=True)
R9,R101=roll(Zc,9),roll(Zc,101)
for w in [61,75,101,121]: ev(f"Zc+r5+r{w}",np.hstack([Z,roll(Zc,5),roll(Zc,w)]))
ev("Zc+r9+r33+r101",np.hstack([Z,R9,roll(Zc,33),R101]))
ang=[c for c in Xdf.columns if c.startswith("angle") or "gravity" in c]
A=Zc[ang]; ev("Zc+r9+r101+ang r201",np.hstack([Z,R9,R101,roll(A,201)]))
X=np.hstack([Z,R9,R101]); sh=KFold(10,shuffle=True,random_state=42)
ev("[섞음] 원본",Xdf.values,sh); ev("[섞음] Zc+r9+r101",X,sh)
