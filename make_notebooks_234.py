import json
def save(cells, path):
    nb={"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":c.strip("\n").splitlines(keepends=True)} for c in cells],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    json.dump(nb,open(path,"w"),ensure_ascii=False,indent=1)

# ================= 2번 HAR =================
save([
'''import numpy as np
import pandas as pd

input_file = "train.csv"
df_train = pd.read_csv(input_file)
df_train.shape''',
'''df_train["subject"].value_counts()''',
'''df_train["Activity"].value_counts()''',
'''from sklearn.preprocessing import OrdinalEncoder

df_train_cat = df_train[["Activity"]]
ordinal_encoder = OrdinalEncoder()
y_train = ordinal_encoder.fit_transform(df_train_cat).flatten()
ordinal_encoder.categories_''',
'''# 특성 엔지니어링 (누수 방지): 교차검증의 각 fold 안에서, 그 fold 에 들어온 행만으로 계산한다
#  같은 사람의 앞뒤 측정 구간 평균 (짧은 창 9, 긴 창 51) 을 원본 특성에 덧붙인다
#  정답(Activity)은 사용하지 않고, 학습 fold 와 시험 fold 의 값이 서로 섞이지 않는다 (subject 는 묶음 기준으로만 쓰고 특성에서는 뺌)
def har_features(X):
    subject = X["subject"]
    num = X.drop(columns=["subject"])
    parts = [num.to_numpy()]
    for w in [9, 51]:
        roll = num.copy()
        for s in subject.unique():
            idx = (subject == s).to_numpy()
            roll.loc[idx] = num.loc[idx].rolling(w, center=True, min_periods=1).mean()
        parts.append(roll.to_numpy())
    return np.hstack(parts)

X_all = df_train.drop(["Activity"], axis=1)          # 561 특성 + subject
X_raw = df_train.drop(["subject", "Activity"], axis=1).to_numpy()
har_features(X_all.iloc[:500]).shape''',
'''# 1) 모델 비교: 원본 특성 vs 내 특성, 여러 분류 모델 (10-fold, 섞지 않음)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "SGD": SGDClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=3000),
    "LinearSVC": LinearSVC(max_iter=10000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVC": SVC(C=10),
}
for name, model in models.items():
    raw = cross_val_score(model, X_raw, y_train, cv=10, scoring="accuracy").mean()
    fe = cross_val_score(make_pipeline(FunctionTransformer(har_features), model), X_all, y_train, cv=10, scoring="accuracy").mean()
    print(f"{name:20s} 원본 {raw:.4f}   내 특성 {fe:.4f}")''',
'''# 2) 하이퍼파라미터 탐색: 가장 좋은 SVC 의 C, gamma (Grid Search, 5-fold)
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(FunctionTransformer(har_features), SVC())
param_grid = {"svc__C": [10, 100, 1000], "svc__gamma": ["scale", 0.0005, 0.002]}
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_all, y_train)
print(grid_search.best_params_, grid_search.best_score_)
for mean, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
    print(f"{mean:.4f}", params)''',
'''# 3) 최종 모델 10-fold 교차 검증 (섞지 않음)
final_clf = grid_search.best_estimator_

cv_res = cross_val_score(final_clf, X_all, y_train, cv=10, scoring="accuracy")
cv_res''',
'''np.mean(cv_res)''',
'''from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(final_clf, X_all, y_train, cv=10)
confusion_matrix(y_train, y_pred)'''],"/home/claude/nb234/mini_proj_02_20221329.ipynb")

# ================= 공통 회귀 셀 =================
display='''def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())'''
lc='''# 학습 곡선: 학습 데이터 양에 따른 학습/검증 RMSE
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

sizes, train_sc, val_sc = learning_curve(%s, %s, %s, cv=5, scoring="neg_mean_squared_error",
                                         train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
plt.plot(sizes, np.sqrt(-train_sc.mean(axis=1)), "r-+", label="train")
plt.plot(sizes, np.sqrt(-val_sc.mean(axis=1)), "b-", label="validation")
plt.xlabel("Training set size"); plt.ylabel("RMSE"); plt.legend(); plt.show()'''

# ================= 3번 집값 =================
save([
'''import numpy as np
import pandas as pd

input_file = "kc_house_data.csv"
df = pd.read_csv(input_file)
df.info()''',
'''def make_features(df):
    X = df.drop(["id", "price", "date"], axis=1).copy()   # id 는 집 번호라 필요 없는 특성으로 제거
    d = pd.to_datetime(df["date"].str[:8])                 # date 는 숫자로 변환해 사용
    X["sale_year"] = d.dt.year
    X["sale_month"] = d.dt.month
    X["sale_t"] = (d - pd.Timestamp("2014-01-01")).dt.days   # 고정 기준일
    X["age"] = X["sale_year"] - X["yr_built"]
    X["renov"] = (X["yr_renovated"] > 0).astype(int)
    X["since_renov"] = np.where(X["yr_renovated"] > 0, X["sale_year"] - X["yr_renovated"], X["age"])
    X["rot1"] = X["lat"] + X["long"]
    X["rot2"] = X["lat"] - X["long"]
    for name, (la, lo) in {"seattle": (47.6062, -122.3321), "bellevue": (47.6101, -122.2015)}.items():
        X["dist_" + name] = np.sqrt((X["lat"] - la) ** 2 + ((X["long"] - lo) * 0.67) ** 2)
    for a in [15, 30, 60, 75]:
        r = np.deg2rad(a)
        X["rot%d" % a] = X["lat"] * np.cos(r) + X["long"] * np.sin(r)
    X["liv_ratio15"] = X["sqft_living"] / X["sqft_living15"]
    X["lot_ratio15"] = X["sqft_lot"] / X["sqft_lot15"]
    X["liv_per_lot"] = X["sqft_living"] / X["sqft_lot"]
    return X

housing_base = df.drop(["id", "price", "date"], axis=1)
housing_prepared = make_features(df)
housing_labels = df["price"].copy()
housing_prepared.shape''',
display,
'''# 1) 모델 비교: 원본 특성 vs 내 특성, 여러 회귀 모델 (10-fold RMSE)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def rmse_cv(model, X):
    return np.sqrt(-cross_val_score(model, X, housing_labels, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)).mean()

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGB": XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=1),
    "LGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1, n_jobs=1),
}
for name, model in models.items():
    print(f"{name:18s} 원본 {rmse_cv(model, housing_base):10.0f}   내 특성 {rmse_cv(model, housing_prepared):10.0f}")''',
'''# 2) 하이퍼파라미터 탐색: LGBM (Randomized Search, 5-fold)
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.03, 0.05],
                  "colsample_bytree": [0.4, 0.6, 0.8], "subsample": [0.7, 0.8, 1.0]}
rnd_search = RandomizedSearchCV(LGBMRegressor(n_estimators=1500, subsample_freq=1, verbose=-1, n_jobs=1),
                                param_distribs, n_iter=10, cv=5, scoring="neg_mean_squared_error",
                                random_state=42, n_jobs=-1)
rnd_search.fit(housing_prepared, housing_labels)
print(rnd_search.best_params_, np.sqrt(-rnd_search.best_score_))
for mean, params in zip(rnd_search.cv_results_["mean_test_score"], rnd_search.cv_results_["params"]):
    print(f"{np.sqrt(-mean):10.0f}", params)''',
'''# 3) 최종 모델: XGB·LGBM (원래 가격 / log 가격) 4개를 Stacking
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor

def xgb():
    return XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.6, n_jobs=1)

def lgbm():   # Randomized Search 로 찾은 하이퍼파라미터 사용
    return LGBMRegressor(n_estimators=1500, subsample_freq=1, verbose=-1, n_jobs=1, **rnd_search.best_params_)

def log_target(model):
    return TransformedTargetRegressor(model, func=np.log, inverse_func=np.exp)

final_reg = StackingRegressor(
    estimators=[("xgb", xgb()), ("xgb_log", log_target(xgb())), ("lgbm", lgbm()), ("lgbm_log", log_target(lgbm()))],
    final_estimator=LinearRegression(positive=True), cv=5)''',
'''final_scores = cross_val_score(final_reg, housing_prepared, housing_labels,
                               scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
final_rmse_scores = np.sqrt(-final_scores)
display_scores(final_rmse_scores)''',
'''np.mean(final_rmse_scores)''',
lc % ("LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=1)", "housing_prepared", "housing_labels")],
"/home/claude/nb234/mini_proj_03_20221329.ipynb")

# ================= 4번 와인 =================
save([
'''import numpy as np
import pandas as pd

input_file = "winequality-red.csv"
df = pd.read_csv(input_file)
df.shape''',
'''def make_features(df):
    X = df.drop(["quality"], axis=1).copy()
    X["so2_ratio"] = X["free sulfur dioxide"] / X["total sulfur dioxide"]
    X["alc_sulph"] = X["alcohol"] * X["sulphates"]
    X["alc_va"] = X["alcohol"] / X["volatile acidity"]
    X["log_sulph"] = np.log(X["sulphates"])
    X["log_chl"] = np.log(X["chlorides"])
    X["log_tso2"] = np.log(X["total sulfur dioxide"])
    X["acid_total"] = X["fixed acidity"] + X["volatile acidity"] + X["citric acid"]
    return X

wine_base = df.drop(["quality"], axis=1)
wine_prepared = make_features(df)
wine_labels = df["quality"].copy()
wine_prepared.shape''',
display,
'''# 1) 모델 비교: 원본 특성 vs 내 특성, 여러 회귀 모델 (10-fold RMSE)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

def rmse_cv(model, X):
    return np.sqrt(-cross_val_score(model, X, wine_labels, scoring="neg_mean_squared_error", cv=10)).mean()

models = {
    "LinearRegression": LinearRegression(),
    "SGD": make_pipeline(StandardScaler(), SGDRegressor(random_state=42)),
    "SVR": make_pipeline(StandardScaler(), SVR()),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}
for name, model in models.items():
    print(f"{name:18s} 원본 {rmse_cv(model, wine_base):.4f}   내 특성 {rmse_cv(model, wine_prepared):.4f}")''',
'''# 2) 하이퍼파라미터 탐색 (Grid Search, 10-fold): SVR 과 RandomForest
from sklearn.model_selection import GridSearchCV

svr_search = GridSearchCV(make_pipeline(StandardScaler(), SVR()),
                          {"svr__C": [0.3, 1, 3], "svr__gamma": [0.01, 0.03, "scale"], "svr__epsilon": [0.1, 0.3]},
                          cv=10, scoring="neg_mean_squared_error", n_jobs=-1)
svr_search.fit(wine_prepared, wine_labels)
print("SVR", svr_search.best_params_, np.sqrt(-svr_search.best_score_))

rf_search = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=42),
                         {"max_features": [0.33, 0.5, 1.0], "min_samples_leaf": [1, 2, 5]},
                         cv=10, scoring="neg_mean_squared_error", n_jobs=-1)
rf_search.fit(wine_prepared, wine_labels)
print("RF ", rf_search.best_params_, np.sqrt(-rf_search.best_score_))''',
'''# 3) 최종 모델: 튜닝한 SVR·RandomForest + ExtraTrees 를 Voting
from sklearn.ensemble import VotingRegressor

final_reg = VotingRegressor([
    ("svr", svr_search.best_estimator_),
    ("rf", RandomForestRegressor(n_estimators=500, random_state=42, **rf_search.best_params_)),
    ("et", ExtraTreesRegressor(n_estimators=800, max_features=0.5, min_samples_leaf=2, random_state=42))])''',
'''final_scores = cross_val_score(final_reg, wine_prepared, wine_labels, scoring="neg_mean_squared_error", cv=10)
final_rmse_scores = np.sqrt(-final_scores)
display_scores(final_rmse_scores)''',
'''np.mean(final_rmse_scores)''',
lc % ("final_reg", "wine_prepared", "wine_labels")],
"/home/claude/nb234/mini_proj_04_20221329.ipynb")
print("ok")
