import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, VotingRegressor
from lightgbm import LGBMRegressor
df=pd.read_csv('/home/claude/ml-mini-project/data/project04/winequality-red.csv'); y=df["quality"].values
def fe(df):
    X=df.drop(columns=["quality"]).copy()
    X["so2_ratio"]=X["free sulfur dioxide"]/X["total sulfur dioxide"]
    X["alc_sulph"]=X["alcohol"]*X["sulphates"]; X["alc_va"]=X["alcohol"]/X["volatile acidity"]
    X["log_sulph"]=np.log(X["sulphates"]); X["log_chl"]=np.log(X["chlorides"]); X["log_tso2"]=np.log(X["total sulfur dioxide"])
    X["acid_total"]=X["fixed acidity"]+X["volatile acidity"]+X["citric acid"]
    return X
X0=df.drop(columns=["quality"]); X1=fe(df)
def r(m,X,n):
    t=time.time(); s=np.sqrt(-cross_val_score(m,X,y,cv=10,scoring="neg_mean_squared_error",n_jobs=2)); print(f"{n:30s} {s.mean():.4f}  {time.time()-t:.0f}s",flush=True)
svr=lambda C=1,g="scale",e=0.1: make_pipeline(StandardScaler(),SVR(C=C,gamma=g,epsilon=e))
for C,g in [(0.5,"scale"),(1,0.03),(2,0.02),(1,0.05)]: r(svr(C,g),X1,f"SVR 특성 C={C} g={g}")
r(RandomForestRegressor(500,max_features=0.33,min_samples_leaf=2,random_state=42,n_jobs=1),X1,"RF 특성")
r(ExtraTreesRegressor(800,max_features=0.5,min_samples_leaf=2,random_state=42,n_jobs=1),X1,"ET 특성")
r(make_pipeline(StandardScaler(),Ridge(alpha=10)),X1,"Ridge 특성")
est=[("svr",svr(1,0.03)),("rf",RandomForestRegressor(500,max_features=0.33,min_samples_leaf=2,random_state=42,n_jobs=1)),
     ("et",ExtraTreesRegressor(800,max_features=0.5,min_samples_leaf=2,random_state=42,n_jobs=1)),("ridge",make_pipeline(StandardScaler(),Ridge(alpha=10)))]
r(VotingRegressor(est),X1,"Voting 4개")
r(StackingRegressor(est,final_estimator=LinearRegression(positive=True),cv=5),X1,"Stacking 4개")
