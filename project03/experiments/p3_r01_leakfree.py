# 3번 집값 누수 없는 1차 연구: 바깥 KFold(10), 특성은 행 하나에서만(상수 기준점), 튜닝·군집은 fold 안에서
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
df = pd.read_csv('data/project03/kc_house_data.csv'); y = df['price'].values
def make_features(df):
    X = df.drop(["id", "price", "date"], axis=1).copy(); d = pd.to_datetime(df["date"].str[:8])
    X["sale_year"] = d.dt.year; X["sale_month"] = d.dt.month; X["sale_t"] = (d - pd.Timestamp("2014-01-01")).dt.days
    X["age"] = X["sale_year"] - X["yr_built"]; X["renov"] = (X["yr_renovated"] > 0).astype(int)
    X["since_renov"] = np.where(X["yr_renovated"] > 0, X["sale_year"] - X["yr_renovated"], X["age"])
    X["rot1"] = X["lat"] + X["long"]; X["rot2"] = X["lat"] - X["long"]
    for n, (la, lo) in {"sea": (47.6062, -122.3321), "bel": (47.6101, -122.2015)}.items():
        X["d_" + n] = np.sqrt((X["lat"] - la) ** 2 + ((X["long"] - lo) * 0.67) ** 2)
    for a in [15, 30, 60, 75]:
        r = np.deg2rad(a); X[f"r{a}"] = X["lat"] * np.cos(r) + X["long"] * np.sin(r)
    X["liv_ratio15"] = X["sqft_living"] / X["sqft_living15"]; X["lot_ratio15"] = X["sqft_lot"] / X["sqft_lot15"]
    X["liv_per_lot"] = X["sqft_living"] / X["sqft_lot"]
    return X
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """위도·경도 KMeans 중심과의 RBF 유사도 (W03/핸즈온 2장 방식). fold 의 학습 행으로만 fit, 정답 미사용"""
    def __init__(self, n_clusters=30, gamma=200., cols=("lat", "long")):
        self.n_clusters, self.gamma, self.cols = n_clusters, gamma, cols
    def fit(self, X, y=None):
        self.km_ = KMeans(self.n_clusters, n_init=4, random_state=42).fit(X[list(self.cols)].to_numpy()); return self
    def transform(self, X):
        sim = rbf_kernel(X[list(self.cols)].to_numpy(), self.km_.cluster_centers_, gamma=self.gamma)
        return np.hstack([X.to_numpy(), sim])
X = make_features(df); outer = KFold(n_splits=10)
T = lambda m: TransformedTargetRegressor(m, func=np.log, inverse_func=np.exp)
lgb = lambda **k: LGBMRegressor(**{**dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1, n_jobs=1), **k})
def ev(name, m, Xu=X):
    t = time.time(); sc = np.sqrt(-cross_val_score(m, Xu, y, cv=outer, scoring='neg_mean_squared_error', n_jobs=2))
    print(f"| {name} | {sc.mean():,.0f} | {sc.std():,.0f} | {time.time()-t:.0f}s |", flush=True)
print("| 조합 | 10-fold RMSE | fold 표준편차 | 시간 |\n|---|---|---|---|")
part = sys.argv[1] if len(sys.argv) > 1 else 'all'
if part in ('all', 'A'):
    ev("LGBM log 고정 설정 (기준)", T(lgb()))
    for k, g in [(30, 200.), (60, 500.), (100, 1000.)]:
        ev(f"KMeans{k} 유사도(γ={g:.0f}) + LGBM log", make_pipeline(ClusterSimilarity(k, g), T(lgb())))
if part in ('all', 'B'):  # 노트북의 '전체 데이터로 RandomizedSearch → 그 설정으로 10-fold' 대신 중첩 탐색
    dist = {"regressor__num_leaves": [15, 31, 63], "regressor__learning_rate": [0.01, 0.03, 0.05],
            "regressor__colsample_bytree": [0.4, 0.6, 0.8], "regressor__subsample": [0.7, 0.8, 1.0]}
    ev("중첩 RandomizedSearch(10회, 내부 KFold5) LGBM log",
       RandomizedSearchCV(T(lgb()), dist, n_iter=10, cv=KFold(5), scoring='neg_mean_squared_error', random_state=42, n_jobs=1))
