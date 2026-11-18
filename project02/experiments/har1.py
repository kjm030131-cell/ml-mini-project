import pandas as pd, numpy as np, time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
d=pd.read_csv('/home/claude/ml-mini-project/data/project02/train.csv.gz')
X=d.drop(columns=['subject','Activity']).values; y=d['Activity'].values
for name,m in [("LR C=10",LogisticRegression(C=10,max_iter=3000)),("SVC C=10",SVC(C=10)),("SVC C=100",SVC(C=100))]:
    t=time.time(); s=cross_val_score(m,X,y,cv=10,n_jobs=2); print(f"{name:10s} {s.mean():.4f} folds {np.round(s,3)} {time.time()-t:.0f}s",flush=True)
