# 2번 엄격 3차: 특성 묶음별 비중 (앉기↔서기는 몸 기울기 = 중력·각도 특성, 걷기↔계단은 몸통 가속 주파수)
import time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
d = pd.read_csv('data/project02/train.csv.gz'); y = d['Activity'].values
F = d.drop(columns=['subject', 'Activity']); cols = F.columns; X0 = F.to_numpy(); outer = KFold(10)
grp = {"grav": cols.str.startswith('tGravityAcc'), "angle": cols.str.startswith('angle'),
       "t": cols.str.startswith('t'), "f": cols.str.startswith('f')}
def ev(name, w):
    t = time.time(); sc = cross_val_score(SVC(C=100), X0 * w, y, cv=outer, n_jobs=1)
    print(f"| {name} | {sc.mean():.4f} | {sc.min():.3f} | {time.time()-t:.0f}s |", flush=True)
print("| 조합 | 10-fold 평균 | 최저 fold | 시간 |\n|---|---|---|---|")
for g in ["grav", "angle"]:
    for k in [2, 4]:
        w = np.ones(len(cols)); w[grp[g]] = k; ev(f"[S] {g} x{k}", w)
w = np.ones(len(cols)); w[grp["grav"] | grp["angle"]] = 3; ev("[S] grav+angle x3", w)
for g, k in [("f", 0.5), ("t", 0.5)]:
    w = np.ones(len(cols)); w[grp[g]] = k; ev(f"[S] {g} x{k}", w)
