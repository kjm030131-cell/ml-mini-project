import pandas as pd, numpy as np, time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
d=pd.read_csv('/home/claude/ml-mini-project/data/project02/train.csv.gz')
Xdf=d.drop(columns=['subject','Activity']); y=d['Activity'].values; g=d['subject']
roll=lambda df,w: df.groupby(g).transform(lambda s: s.rolling(w,center=True,min_periods=1).mean()).values
Zc=Xdf-Xdf.groupby(g).transform('mean')
RC={w:roll(Zc,w) for w in [5,9,15,25,51,75]}
def ev(name,X,m=None):
    m=m or SVC(C=100); s=cross_val_score(m,X,y,cv=10,n_jobs=2); print(f"{name:34s} {s.mean():.4f} min {s.min():.3f}",flush=True); return s.mean()
Z=Zc.values
for ws in [(9,),(51,),(5,51),(9,25),(15,51),(9,75),(9,25,51),(5,15,51)]:
    ev("Zc+"+"+".join(f"r{w}" for w in ws),np.hstack([Z]+[RC[w] for w in ws]))
X=np.hstack([Z,RC[9],RC[51]])
for C in [10,30,300,1000]: ev(f"best C={C}",X,SVC(C=C))
for gm in [0.5,2]: ev(f"best gamma x{gm}",X,SVC(C=100,gamma=gm/(X.shape[1]*X.var())))
ev("best LR C=1",X,LogisticRegression(C=1,max_iter=5000))
ev("r9+r51 only (no Z)",np.hstack([RC[9],RC[51]]))
