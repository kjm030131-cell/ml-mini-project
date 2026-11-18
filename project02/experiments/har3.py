import pandas as pd, numpy as np, time, sys
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
d=pd.read_csv('/home/claude/ml-mini-project/data/project02/train.csv.gz')
Xdf=d.drop(columns=['subject','Activity']); y=d['Activity'].values; X0=Xdf.values
g=d['subject']
def roll(w, f="mean", by_subj=True):
    if by_subj: return Xdf.groupby(g).transform(lambda s: getattr(s.rolling(w,center=True,min_periods=1),f)()).values
    return getattr(Xdf.rolling(w,center=True,min_periods=1),f)().values
def ev(name,X,m):
    t=time.time(); s=cross_val_score(m,X,y,cv=10,n_jobs=2); print(f"{name:40s} {s.mean():.4f} min {s.min():.3f} {time.time()-t:.0f}s",flush=True); return s.mean()
R={w:roll(w) for w in [9,15,21,31,51]}
ev("orig+roll9 (subject 경계)",np.hstack([X0,R[9]]),SVC(C=100))
for w in [21,31,51]: ev(f"orig+roll{w}",np.hstack([X0,R[w]]),SVC(C=100))
ev("orig+roll9+roll31",np.hstack([X0,R[9],R[31]]),SVC(C=100))
ev("orig+roll9+roll51",np.hstack([X0,R[9],R[51]]),SVC(C=100))
S=roll(15,"std"); ev("orig+roll15+std15",np.hstack([X0,R[15],S]),SVC(C=100))
for C in [10,1000]: ev(f"orig+roll15 C={C}",np.hstack([X0,R[15]]),SVC(C=C))
ev("orig+roll15 LR C=10",np.hstack([X0,R[15]]),LogisticRegression(C=10,max_iter=5000))
