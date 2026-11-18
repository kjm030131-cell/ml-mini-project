import pandas as pd, numpy as np, time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
df=pd.read_csv('/home/claude/ml-mini-project/data/project03/kc_house_data.csv')
y=df["price"].values
def base(df): return df.drop(columns=["price","date","id"])
def fe(df):
    X=base(df).copy(); d=pd.to_datetime(df["date"].str[:8])
    X["sale_year"]=d.dt.year; X["sale_month"]=d.dt.month; X["sale_t"]=(d-d.min()).dt.days
    X["age"]=X["sale_year"]-X["yr_built"]; X["renov"]=(X["yr_renovated"]>0).astype(int)
    X["since_renov"]=np.where(X["yr_renovated"]>0,X["sale_year"]-X["yr_renovated"],X["age"])
    X["rot1"]=X["lat"]+X["long"]; X["rot2"]=X["lat"]-X["long"]
    return X
def rmse(m,X,name):
    t=time.time(); s=np.sqrt(-cross_val_score(m,X,y,cv=10,scoring="neg_mean_squared_error",n_jobs=1)); print(f"{name:34s} {s.mean():10.0f}  {time.time()-t:.0f}s",flush=True)
rmse(LinearRegression(),df.drop(columns=["price","date"]),"슬라이드 LR(id 포함)")
X0=base(df); X1=fe(df)
rmse(LGBMRegressor(n_estimators=2000,learning_rate=0.02,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1,n_jobs=1),X0,"LGBM 기본특성")
rmse(LGBMRegressor(n_estimators=2000,learning_rate=0.02,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1,n_jobs=1),X1,"LGBM 날짜·나이 특성")
rmse(XGBRegressor(n_estimators=2000,learning_rate=0.02,max_depth=6,subsample=0.8,colsample_bytree=0.7,n_jobs=1),X1,"XGB 날짜·나이 특성")
rmse(RandomForestRegressor(n_estimators=300,n_jobs=1,random_state=42),X1,"RF")
