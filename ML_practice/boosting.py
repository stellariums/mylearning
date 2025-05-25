#X.Wan,Z.Zhang,et al.J.Phys.Chem.Lett.2021,12,6111
#预测CO2RR

#adaptive boosting
#串行计算，增加预测不好的点的权重

import pandas as pd
df=pd.read_csv("CO2RR.csv",header=None)#数据没有列标签

#归一化
from sklearn.preprocessing import MinMaxScaler
X=df.drop(20,axis=1).values#丢掉最后一行（结果）
mms=MinMaxScaler()#归一化器
X=mms.fit_transform(X)
y=df[20].values

from sklearn.model_selection import train_test_split#切分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=16)#随机模式为16
X_test.shape,X_train.shape

from sklearn.ensemble import AdaBoostRegressor
adbr=AdaBoostRegressor()
adbr.fit(X_train,y_train)
y_train_pred=adbr.predict(X_train)
y_test_pred=adbr.predict(X_test)

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
rmse_train=np.sqrt(mse(y_train,y_train_pred))
r2_train=r2_score(y_train,y_train_pred)

rmse_train=np.sqrt(mse(y_test,y_test_pred))
r2_train=r2_score(y_test,y_test_pred)

#图
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(y_train,y_train_pred,s=100,color="w",edgecolors="b",marker="s")
plt.scatter(y_test,y_test_pred,s=100,color="w",edgecolors="r",marker="o")#改变几何形状（三角形）
plt.plot([0.25,2.25],[0.25,2.25],"k--")
plt.xlim(0.25,2.25)
plt.ylim(0.25,2.25)
plt.show()

#梯度提升回归
#Gradient Boosting
#对于训练集效果很好，测试集比较差，会过拟合

from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)

y_train_pred=gbr.predict(X_train)
y_test_pred=gbr.predict(X_test)

rmse_train=np.sqrt(mse(y_train,y_train_pred))
r2_train=r2_score(y_train,y_train_pred)

rmse_train=np.sqrt(mse(y_test,y_test_pred))
r2_train=r2_score(y_test,y_test_pred)
#0.008VS0.19
#0.999VS0.49

#网格搜索方法
#调整超参数
#一种一种来，减小计算量
from sklearn.model_selection import GridSearchCV
#网格加交叉验证
parameters={
    "n_estimators":list(range(50,650,50))#字典
}

gbr2=GradientBoostingRegressor()
gs=GridSearchCV(gbr2,parameters,cv=40,scoring="neg_mean_squared_error")#留一法不能使用R2打分
gs.fit(X,y)

#结果
gs.best_params_#输出最佳
gs.best_score_#输出最佳分数
np.sqrt(-gs.best_score_)

parameters={
    "n_estimators":[300],
    "max_depth":[3,4,5,6,7]
}

gbr2=GradientBoostingRegressor()
gs=GridSearchCV(gbr2,parameters,cv=40,scoring="neg_mean_squared_error")#留一法不能使用R2打分
gs.fit(X,y)

#结果
gs.best_params_#输出最佳
gs.best_score_#输出最佳分数
np.sqrt(-gs.best_score_)

parameters={
    "n_estimators":[300],
    "max_depth":[6],
    "min_samples_split":[2,3,4,5,6,7]
}

gbr3=gs.best_estimator_#取出最优模型
gbr3.fit(X_train,y_train)

params={
    "n_estimators":[500],#"树的数量"
    "max_depth":[5],#"树的最大深度"
    "min_samples_split":[5],#最小样本分割
    "learning_rate":[0.005],#学习率
    "loss":"huber"#损失函数
}

gbr4=GradientBoostingRegressor(**params)
gbr4.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
scores_rmse=cross_val_score(gbr4,X,y,scoring="neg_root_mean_squared_error",cv=len(X))
from sklearn.model_selection import cross_val_predict
y_cv=cross_val_predict(gbr4,X,y,cv=40)

plt.figure(figsize=(4,4))
plt.scatter(y,y_cv,s=100,color="w",edgecolors="b")
plt.plot([0.25,2.25],[0.25,2.25],"k--")
plt.xlim(0.25,2.25)
plt.ylim(0.25,2.25)
plt.show()

#GradientBoosting泛化能力十分强
#若刚开始就不好，调参后大概率也不会很好

#泛化误差=偏差（Boosting）（经验上结果更好）+方差（Bagging）+噪声

#若两者结合
#MultiBoosting(Webb,2000) 先adaboosting后bagging 效果好，但是时间成本上升快
#Iterative Bagging(Breiman,2001) 先bagging后boosting 效果不算好，是两者的折中

#XGBoost
#极致梯度提升（陈天奇 2014）
#可解决超billions样本问题
#在损失函数中加入树复杂度项
#引入二阶项
#是GBDT的高效体现，效果很好

from xgboost import XGBRegressor
clf=XGBRegressor()
clf.fit(X,y)