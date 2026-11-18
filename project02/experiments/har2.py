import pandas as pd, numpy as np, time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
d=pd.read_csv('/home/claude/ml-mini-project/data/project02/train.csv.gz')
Xdf=d.drop(columns=['subject','Activity']); y=d['Activity'].values
for w in [3,5,9,15]:
    R=Xdf.rolling(w,center=True,min_periods=1).mean().values
    for name,X in [("roll",R),("orig+roll",np.hstack([Xdf.values,R]))]:
        s=cross_val_score(SVC(C=100),X,y,cv=10,n_jobs=2); print(f"w={w:2d} {name:9s} {s.mean():.4f} min {s.min():.3f}",flush=True)
