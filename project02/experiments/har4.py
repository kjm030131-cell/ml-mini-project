import pandas as pd, numpy as np, time
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
d=pd.read_csv('/home/claude/ml-mini-project/data/project02/train.csv.gz')
Xdf=d.drop(columns=['subject','Activity']); y=d['Activity'].values; X0=Xdf.values; g=d['subject']
roll=lambda df,w: df.groupby(g).transform(lambda s: s.rolling(w,center=True,min_periods=1).mean()).values
def ev(name,X,m=None):
    m=m or SVC(C=100); t=time.time(); s=cross_val_score(m,X,y,cv=10,n_jobs=2); print(f"{name:34s} {s.mean():.4f} min {s.min():.3f} {np.round(s,3)}",flush=True); return s.mean()
Zc=(Xdf-Xdf.groupby(g).transform('mean'))           # 사람별 평균 빼기
Zs=Zc/(Xdf.groupby(g).transform('std')+1e-6)         # 사람별 표준화
ev("subj-center",Zc.values); ev("orig+subj-center",np.hstack([X0,Zc.values]))
ev("subj-z",Zs.values)
ev("orig+roll9+roll51+subjcenter",np.hstack([X0,roll(Xdf,9),roll(Xdf,51),Zc.values]))
ev("subjcenter+roll(subjcenter)9,51",np.hstack([Zc.values,roll(Zc,9),roll(Zc,51)]))
p=cross_val_predict(SVC(C=100),np.hstack([X0,roll(Xdf,9),roll(Xdf,51)]),y,cv=10,n_jobs=2)
print(pd.crosstab(y,p))
