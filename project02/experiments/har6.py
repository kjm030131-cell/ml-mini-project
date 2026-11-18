import pandas as pd, numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
d=pd.read_csv('/home/claude/ml-mini-project/data/project02/train.csv.gz')
Xdf=d.drop(columns=['subject','Activity']); y=d['Activity'].values; g=d['subject']
roll=lambda df,w,f="mean": df.groupby(g).transform(lambda s: getattr(s.rolling(w,center=True,min_periods=1),f)()).values
Zc=Xdf-Xdf.groupby(g).transform('mean'); Z=Zc.values
def ev(name,X,m=None):
    m=m or SVC(C=100); s=cross_val_score(m,X,y,cv=10,n_jobs=2); print(f"{name:30s} {s.mean():.4f} min {s.min():.3f}",flush=True); return s.mean()
R9=roll(Zc,9)
for w in [101,151]: ev(f"Zc+r9+r{w}",np.hstack([Z,R9,roll(Zc,w)]))
M9=roll(Zc,9,"median"); M75=roll(Zc,75,"median")
ev("Zc+med9+med75",np.hstack([Z,M9,M75]))
R75=roll(Zc,75); X=np.hstack([Z,R9,R75])
ev("Zc+r9+r75 LGBM",X,LGBMClassifier(n_estimators=400,learning_rate=0.05,colsample_bytree=0.3,verbose=-1,n_jobs=1))
ev("Zc+r9+r75 soft vote SVC+LR",X,VotingClassifier([("s",SVC(C=100,probability=True)),("l",LogisticRegression(C=1,max_iter=5000))],voting="soft"))
p=cross_val_predict(SVC(C=100),X,y,cv=10,n_jobs=2); w=np.where(p!=y)[0]
print("오답",len(w)); print(pd.Series([f"{a}->{b}" for a,b in zip(y[w],p[w])]).value_counts().head(8))
print("오답 위치(행)", w[:60])
