import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
df=pd.read_csv('/home/claude/ml-mini-project/data/project03/kc_house_data.csv'); y=df["price"].values
def fe(df, geo=True):
    X=df.drop(columns=["price","date","id"]).copy(); d=pd.to_datetime(df["date"].str[:8])
    X["sale_year"]=d.dt.year; X["sale_month"]=d.dt.month; X["sale_t"]=(d-d.min()).dt.days
    X["age"]=X["sale_year"]-X["yr_built"]; X["renov"]=(X["yr_renovated"]>0).astype(int)
    X["since_renov"]=np.where(X["yr_renovated"]>0,X["sale_year"]-X["yr_renovated"],X["age"])
    X["rot1"]=X["lat"]+X["long"]; X["rot2"]=X["lat"]-X["long"]
    if geo:
        for nm,(la,lo) in {"sea":(47.6062,-122.3321),"bel":(47.6101,-122.2015)}.items():
            X["d_"+nm]=np.sqrt((X["lat"]-la)**2+((X["long"]-lo)*0.67)**2)
        for a in [15,30,60,75]:
            r=np.deg2rad(a); X[f"r{a}"]=X["lat"]*np.cos(r)+X["long"]*np.sin(r)
        X["liv_ratio15"]=X["sqft_living"]/X["sqft_living15"]; X["lot_ratio15"]=X["sqft_lot"]/X["sqft_lot15"]
        X["liv_per_lot"]=X["sqft_living"]/X["sqft_lot"]
    return X
def rmse(m,X,name):
    t=time.time(); s=np.sqrt(-cross_val_score(m,X,y,cv=10,scoring="neg_mean_squared_error",n_jobs=1)); print(f"{name:34s} {s.mean():10.0f}  {time.time()-t:.0f}s",flush=True)
X1=fe(df,False); X2=fe(df,True)
xgb=lambda: XGBRegressor(n_estimators=3000,learning_rate=0.01,max_depth=6,subsample=0.8,colsample_bytree=0.6,min_child_weight=1,n_jobs=1)
lgb=lambda: LGBMRegressor(n_estimators=4000,learning_rate=0.01,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,verbose=-1,n_jobs=1)
rmse(xgb(),X2,"XGB 위치특성")
rmse(TransformedTargetRegressor(xgb(),func=np.log,inverse_func=np.exp),X2,"XGB 위치특성 log(price)")
rmse(lgb(),X2,"LGBM 위치특성")
rmse(TransformedTargetRegressor(lgb(),func=np.log,inverse_func=np.exp),X2,"LGBM 위치특성 log(price)")
