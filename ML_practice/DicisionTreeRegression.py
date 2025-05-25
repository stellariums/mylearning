#J.zheng,x.sun,et.al.j.phys.chem.c.2020,124,13695-13705

import scipy.io as scio#由于作者采用mat格式储存数据，需转换
data=scio.loadmat("HER3.mat")
import pandas as pd
df=pd.DataFrame(data=["input"])
df["dG(H)"]=data["output"][0]#二维数组
#可以使用reshape(-1)

#pearson系数查看
X=data["input"]
y=data["output"][0]
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
import numpy as np

dt=DecisionTreeRegressor()
dt.fit(X,y)
rmse=np.sqrt(mse(y_true=y,y_pred=dt.predict(X)))
print(f"Training R2={dt.score(X,y):.3f}")
print(f"Training rmse={rmse:.3f}")
#结果R2=1,RMSE=0
#过拟合了

#画图
import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
plt.scatter(y,dt.predict(X),s=80,edgecolors="k",alpha=0.7)#alpha为透明度
plt.plot([-3,3],[-3,3],"r--")#画对角直线
plt.show()

#添加限定深度
dt.get_depth()#获取当前深度
#结果为14

dt2=DecisionTreeRegressor(max_depth=5)
#R2=0.977 rmse=0.145
plt.figure(figsize=(4,4))
plt.scatter(y,dt2.predict(X),c="b",s=80,edgecolors="k",alpha=0.7)#alpha为透明度,c为颜色
plt.plot([-3,3],[-3,3],"r--")#画对角直线
plt.show()

#交叉验证
from sklearn.model_selection import KFold,cross_val_score
crossvalidation=KFold(n_splits=10,shuffle=True)#构建分割器，折数为10，打乱
r2_scores=cross_val_score(dt,X,y,scoring="r2",cv=crossvalidation)
rmse_scores=cross_val_score(dt,X,y,scoring="neg_root_mean_squared_error",cv=crossvalidation)

crossvalidation=KFold(n_splits=10,shuffle=True)#构建分割器，折数为10，打乱
r2_scores=cross_val_score(dt2,X,y,scoring="r2",cv=crossvalidation)
rmse_scores=cross_val_score(dt2,X,y,scoring="neg_root_mean_squared_error",cv=crossvalidation)

#限定后泛化性更好

#可视化
#利用交叉验证和数据集构建图
from sklearn.model_selection import cross_val_predict
y_cv=cross_val_predict(dt,X,y,cv=crossvalidation)
plt.figure(figsize=(4,4))
plt.scatter(y,y_cv,c="b",s=80,edgecolors="k",alpha=0.7)#alpha为透明度,c为颜色
plt.plot([-3,3],[-3,3],"r--")#画对角直线
plt.show()

y_cv2=cross_val_predict(dt2,X,y,cv=crossvalidation)
plt.figure(figsize=(4,4))
plt.scatter(y,y_cv2,s=80,edgecolors="k",alpha=0.7)#alpha为透明度,c为颜色
plt.plot([-3,3],[-3,3],"r--")#画对角直线
plt.show()

#随机森林
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X,y)
rmse=np.sqrt(mse(y_true=y,y_pred=rf.predict(X)))
print(f"Training R2={dt.score(X,y):.3f}")
print(f"Training rmse={rmse:.3f}")

plt.figure(figsize=(4,4))
plt.scatter(y,rf.predict(X),s=80,edgecolors="k",alpha=0.7)#alpha为透明度
plt.plot([-3,3],[-3,3],"r--")#画对角直线
plt.show()

y_cv=cross_val_predict(rf,X,y,cv=crossvalidation)
plt.figure(figsize=(4,4))
plt.scatter(y,y_cv,c="b",s=80,edgecolors="k",alpha=0.7)#alpha为透明度,c为颜色
plt.plot([-3,3],[-3,3],"r--")#画对角直线
plt.show()

#调整树木个数
scores_rmse_test=[]
for i in range(1,101):
    rf_test=RandomForestRegressor(n_estimators=i)
    rmse_scores=-cross_val_score(rf_test,X,y,scoring="neg_root_mean_squared_error",cv=crossvalidation,n_jobs=-1)
    #n_jobs指调用CPU的个数
    scores_rmse_test.append(np.mean(rmse_scores))

plt.plot(scores_rmse_test)
plt.show()
#会有波动
#总体上呈指数下降

#无需担心过拟合

#极端树算法
#无采样，在样本中选择随机划分
#单颗树预测结果可能不准确
from sklearn.ensemble import ExtraTreesRegressor
et=ExtraTreesRegressor()
et.fit(X,y)
#可能效果好于随机森林

for i in range(1,101):
    rf_test=ExtraTreesRegressor(n_estimators=i)
    rmse_scores=-cross_val_score(rf_test,X,y,scoring="neg_root_mean_squared_error",cv=crossvalidation,n_jobs=-1)
    #n_jobs指调用CPU的个数
    scores_rmse_test.append(np.mean(rmse_scores))
#总体上呈指数下降

#互信息
#使用信息熵描述
#描述两个量之间的相关程度(X-y)
#MI值越大，依赖程度越高
#永远大于等于0
#为0，说明两变量独立

from sklearn.feature_selection import mutual_info_regression
mi=mutual_info_regression(X=X,y=y)

#特征重要性
#只有树才能用
#描述哪个变量更重要

importances_dt=dt.feature_importances_#获取各个变量的特征值
#已经归一化

#柱状图可视化
imp_sort_dt=np.argsort(importances_dt)[::-1]
#从小到大排序，输出相应的索引值
#-1表示反转，最终变为从大到小

fea_name=df.columns.values[:-1].astype(str)
#获取列标签，删除最后一个，设为字符串

plt.figure(figsize=(10,5))
plt.bar(x=fea_name[imp_sort_dt][:10],height=importances_dt[imp_sort_dt][:10],edgecolor="k",color="b")
#取前十个
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)#调整字体大小
plt.show()

#随机森林、极端树也可以进行此项操作


#利用shap值解释模型
import shap
#注意版本
#numpy=1.24.4  1.23.5
#numba=0.57.1  0.56.4
#shap=0.42.1
#scipy=1.11.1  1.10.0

shap.initjs()#绘图准备
#使用kernel方法，此为通用
explainer=shap.TreeExplainer(rf)#构建树解释器、专门针对树，更快
shap_values=explainer.shap_values(X)
#132*20
#每行变量都有
shap_values[0]#20个
#20个变量对第一个样本的影响(相对于rf.predict(X)[0])

#使用js可视化
shap.force_plot(explainer.expected_value,shap_values[0],fea_name)#第一个为期望值，看第一个变量
shap.force_plot(explainer.expected_value,shap_values,X)#看所有的变量

shap.summary_plot(shap_values,X)#绘制各变量值的排名
shap.summary_plot(shap_values,X,plot_type="bar")#平均绝对值

shap_interaction_values=explainer.shap_interaction_values(X)#各变量之间的相互作用
#可理解为shap值的皮尔逊相关系数
shap.summary_plot(shap_interaction_values,X)

#依赖图
shap.dependence_plot("Feature 0",shap_values,X)
#shap与feature0之间的关系

#svr
from sklearn.svm import SVR
svr=SVR()
svr.fit(X,y)
#svr.feature_importances 不能这样做
#需采用核方法
explainer_svr=shap.KernelExplainer(svr.predict,X)#但是对于132*20来说太慢了，需采样
explainer_svr=shap.KernelExplainer(svr.predict,shap.sample(X,10))
shap_values_svr=explainer_svr.shap_values(X)

#可视化
shap.force_plot(explainer_svr.expected_value,shap_values_svr,X)
#柱状图
shap.summary_plot(shap_values_svr,X,plot_type="bar")