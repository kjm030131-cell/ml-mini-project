# 2번 HAR 누수 없는 1차 연구: 바깥 KFold(10, 섞지 않음) 객체, 전처리·튜닝은 모두 fold 안에서
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
d = pd.read_csv('data/project02/train.csv.gz')
X_all = d.drop(columns=['Activity']); y = d['Activity'].values
X_raw = d.drop(columns=['subject', 'Activity']).to_numpy()
outer = KFold(n_splits=10)
def roll_feats(X, wins=(9, 51), center=False):
    """같은 사람의 앞뒤 행 평균. 들어온 행(=그 fold)만으로 계산, 정답 미사용"""
    s = X['subject'].to_numpy(); num = X.drop(columns=['subject'])
    parts = [];
    if center: num = num - num.groupby(s).transform('mean')
    parts.append(num.to_numpy())
    for w in wins: parts.append(num.groupby(s).transform(lambda c: c.rolling(w, center=True, min_periods=1).mean()).to_numpy())
    return np.hstack(parts)
def ev(name, m, X):
    t = time.time(); sc = cross_val_score(m, X, y, cv=outer, scoring='accuracy', n_jobs=2)
    print(f"| {name} | {sc.mean():.4f} | {sc.min():.3f} | {time.time()-t:.0f}s |", flush=True)
print("| 조합 | 10-fold 평균 | 최저 fold | 시간 |\n|---|---|---|---|")
part = sys.argv[1] if len(sys.argv) > 1 else 'all'
if part in ('all', 'S'):  # 엄격: 각 행 561 특성만
    ev("[S] SVC C=10 원본", SVC(C=10), X_raw)
    ev("[S] SVC C=100 원본", SVC(C=100), X_raw)
    ev("[S] Scaler+SVC C=10", make_pipeline(StandardScaler(), SVC(C=10)), X_raw)
    ev("[S] LogisticRegression C=1", LogisticRegression(C=1, max_iter=5000), X_raw)
    ev("[S] LinearSVC C=1", LinearSVC(C=1, max_iter=20000), X_raw)
    ev("[S] 중첩 Grid SVC C(1,10,100,1000) 내부 KFold(5)",
       GridSearchCV(SVC(), {"C": [1, 10, 100, 1000]}, cv=KFold(5), n_jobs=1), X_raw)
if part in ('all', 'T'):  # 문맥: 같은 사람 앞뒤 행 (fold 안 행만, 정답 미사용) — 규칙 확인 필요
    ev("[T] roll 9+51 + SVC C=100", make_pipeline(FunctionTransformer(roll_feats), SVC(C=100)), X_all)
    ev("[T] 사람평균빼기 + roll 9+51 + SVC C=100",
       make_pipeline(FunctionTransformer(roll_feats, kw_args={'center': True}), SVC(C=100)), X_all)
    ev("[T] 사람평균빼기 + roll 5+75 + SVC C=100",
       make_pipeline(FunctionTransformer(roll_feats, kw_args={'center': True, 'wins': (5, 75)}), SVC(C=100)), X_all)
