import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
df=pd.read_csv('/home/claude/ml-mini-project/data/project04/winequality-red.csv'); print(df.shape, df.columns.tolist()[:3], df.columns[-1])
X=df.drop(columns=["quality"]); y=df["quality"].values
def r(m,X,n):
    t=time.time(); s=np.sqrt(-cross_val_score(m,X,y,cv=10,scoring="neg_mean_squared_error",n_jobs=2)); print(f"{n:28s} {s.mean():.4f}  {time.time()-t:.0f}s",flush=True)
r(LinearRegression(),X,"LR")
r(make_pipeline(StandardScaler(),SVR(C=1)),X,"SVR C=1")
r(make_pipeline(StandardScaler(),SVR(C=3,gamma=0.05)),X,"SVR C=3 g.05")
r(RandomForestRegressor(500,random_state=42,n_jobs=1),X,"RF")
r(ExtraTreesRegressor(500,random_state=42,n_jobs=1),X,"ET")
r(GradientBoostingRegressor(n_estimators=300,learning_rate=0.03,max_depth=4,subsample=0.8,random_state=42),X,"GBR")
r(LGBMRegressor(n_estimators=500,learning_rate=0.02,num_leaves=15,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1,n_jobs=1),X,"LGBM")
