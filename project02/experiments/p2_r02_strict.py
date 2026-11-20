# 2번 엄격(각 행 561 특성만) 2차: gamma, LGBM, Stacking(내부 KFold), 오차 분석
import time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
d = pd.read_csv('data/project02/train.csv.gz'); y = d['Activity'].values
X = d.drop(columns=['subject', 'Activity']).to_numpy(); outer = KFold(10)
def ev(name, m):
    t = time.time(); sc = cross_val_score(m, X, y, cv=outer, n_jobs=1)
    print(f"| {name} | {sc.mean():.4f} | {sc.min():.3f} | {time.time()-t:.0f}s |", flush=True)
print("| 조합 | 10-fold 평균 | 최저 fold | 시간 |\n|---|---|---|---|")
g0 = 1 / (X.shape[1] * X.var())
for f in [0.3, 3]: ev(f"[S] SVC C=100 gamma {f}배", SVC(C=100, gamma=g0 * f))
lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, colsample_bytree=0.3, verbose=-1, n_jobs=1)
ev("[S] LGBM 500", lgb)
ev("[S] Stacking SVC+LR+LGBM → LR (내부 KFold5)", StackingClassifier(
    [("svc", SVC(C=100)), ("lr", LogisticRegression(C=1, max_iter=5000)), ("lgb", lgb)],
    final_estimator=LogisticRegression(max_iter=5000), cv=KFold(5)))
p = cross_val_predict(SVC(C=100), X, y, cv=outer)
labs = sorted(set(y)); print("\n오차 행렬 (SVC C=100, 행=정답)", labs); print(confusion_matrix(y, p, labels=labs))
