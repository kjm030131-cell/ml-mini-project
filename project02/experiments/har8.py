import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
d=pd.read_csv('/home/claude/ml-mini-project/data/project02/train.csv.gz')
Xdf=d.drop(columns=['subject','Activity']); y=d['Activity'].values; g=d['subject']
roll=lambda df,w: df.groupby(g).transform(lambda s: s.rolling(w,center=True,min_periods=1).mean()).values
Zc=Xdf-Xdf.groupby(g).transform('mean'); Z=Zc.values
def ev(name,X,m):
    s=cross_val_score(m,X,y,cv=10,n_jobs=1); print(f"{name:34s} {s.mean():.4f} min {s.min():.3f}",flush=True)
X=np.hstack([Z,roll(Zc,5),roll(Zc,75)])
for C in [0.1,0.3,3]: ev(f"LR C={C}",X,LogisticRegression(C=C,max_iter=5000))
Xr=np.hstack([Z,roll(Zc,5),roll(Zc,31),roll(Zc,75),roll(Zc,151)])
ev("SVC 4창",Xr,SVC(C=100)); ev("LR 4창 C=1",Xr,LogisticRegression(C=1,max_iter=5000))
