import numpy as np, warnings; warnings.filterwarnings("ignore")
exec(open("p3b.py").read().split("rmse(xgb(),X2")[0])
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
T=lambda m: TransformedTargetRegressor(m,func=np.log,inverse_func=np.exp)
rmse(VotingRegressor([("x",xgb()),("l",T(lgb()))]),X2,"Voting XGB + LGBM(log)")
rmse(VotingRegressor([("x",xgb()),("xl",T(xgb())),("l",lgb()),("ll",T(lgb()))]),X2,"Voting 4개")
rmse(StackingRegressor([("x",xgb()),("xl",T(xgb())),("l",lgb()),("ll",T(lgb()))],final_estimator=LinearRegression(positive=True),cv=5),X2,"Stacking 4개 (LR 판정)")
