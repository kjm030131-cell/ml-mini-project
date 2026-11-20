# 3번 2차: KMeans60 위치 유사도 + Stacking(XGB·XGB log·LGBM·LGBM log → LR positive, 내부 KFold5), zipcode 범주형
import sys, numpy as np
sys.argv = [sys.argv[0], 'none']
exec(open('project03/experiments/p3_r01_leakfree.py', encoding='utf-8').read())
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
xgb = lambda: XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.6, n_jobs=1)
stack = lambda: StackingRegressor([("x", xgb()), ("xl", T(xgb())), ("l", lgb()), ("ll", T(lgb()))],
                                  final_estimator=LinearRegression(positive=True), cv=KFold(5))
ev("XGB log 고정", T(xgb()))
ev("KMeans60 + XGB log", make_pipeline(ClusterSimilarity(60, 500.), T(xgb())))
Xz = X.copy(); Xz["zipcode"] = Xz["zipcode"].astype("category")
ev("zipcode 범주형 + LGBM log", T(lgb()), Xz)
ev("KMeans60 + Stacking 4개 (내부 KFold5)", make_pipeline(ClusterSimilarity(60, 500.), stack()))
