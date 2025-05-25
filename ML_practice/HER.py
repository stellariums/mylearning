import pandas as pd
import sklearn.metrics
df=pd.read_csv("C:/Users/20414/Desktop/data/HER_DATA.csv")

y=df["GH(eV)"].values
X=df.drop(["atom","GH(eV)","Size"],axis=1).values

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)

#系数
coef=lr.coef_#返回一元数组
#截距
intercept=lr.intercept_

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
R2=r2_score(y_true=y,y_pred=lr.predict(X))
mae=mean_absolute_error(y_true=y,y_pred=lr.predict(X))
mse=mean_squared_error(y_true=y,y_pred=lr.predict(X))
#print(f"R2={R2:.3f}")

#预测值VS真实值
import matplotlib.pyplot as plt
# plt.figure(figsize=(5,5))
# plt.scatter(y,lr.predict(X),s=80,edgecolors="k",alpha=0.7)
# plt.plot([-1.8,2.5],[-1.8,2.5],"r--")
# plt.xlim(-1.8,2.5)
# plt.ylim(-1.8,2.5)
# plt.show()

#非负限定演示
#lr2=LinearRegression(positive=True)
#help(LinearRegression)查看提示
#lr2.fit(X,y)
#lr2.coef_

#导入对数函数所在库
#演示广义线性模型

import numpy as np
# lr3=LinearRegression()
# lr3.fit(X,np.log(abs(y)))#y不能为负


df["docc2"]=df["docc"]*df["docc"]
df["q2"]=df["q(e)"]*df["q(e)"]
X2=df.drop(["atom","GH(eV)","Size","q(e)","docc"],axis=1).values
# lr4=LinearRegression()
# lr4.fit(X2,y)
# R=r2_score(y_true=y,y_pred=lr4.predict(X2))
#print(R)

#拆分两集
# X_train,y_train=X[:-20],y[:-20]
# X_test,y_test=X[-20:],y[-20:]
# X_train.shape,y_train.shape
# X_test.shape,y_test.shape
# lr_lo=LinearRegression()
# y_test_pred=lr_lo.fit(X_train,y_train).predict(X_test)

# R2_lo=r2_score(y_test,y_test_pred)
# print(f"R2={R2_lo:.3f}")

#画双图
# plt.figure(figsize=(5,5))
# plt.scatter(y_train,lr.predict(X_train),s=80,edgecolors="k",alpha=0.7)#训练集
# plt.scatter(y_test,y_test_pred,s=80,color="yellow",marker="^",edgecolors="k",alpha=0.7)#测试集
# plt.plot([-1.8,2.5],[-1.8,2.5],"r--")
# plt.xlim(-1.8,2.5)
# plt.ylim(-1.8,2.5)
# plt.show()

#打乱——多次留出
# from sklearn.utils import shuffle
# r2_lo=[]
# for i in range(20):
#     X_r,y_r=shuffle(X2,y)
#     Xr_train,yr_train=X_r[:-20],y_r[:-20]
#     Xr_test,yr_test=X_r[-20:],y_r[-20:]
#     lr5=LinearRegression()
#     yr_test_predict=lr5.fit(Xr_train,yr_train).predict(Xr_test)
#     R2r=r2_score(y_true=yr_test,y_pred=yr_test_predict)
#     r2_lo.append(R2r)
# print(np.mean(r2_lo))

#交叉验证算法
# from sklearn.model_selection import cross_val_score
# r2_scores=cross_val_score(lr,X,y,scoring="r2",cv=5)
# print(f"Folds:{len(r2_scores)},mean r2:{np.mean(r2_scores):.3f}")

#查询scoring有什么参数
# import sklearn
# sklearn.metrics.get_scorer_names()

#留一法
# from sklearn.model_selection import LeaveOneOut
# loo=LeaveOneOut()
# sample_x=[1,2,3,4,5,6]
# for train,test in loo.split(sample_x):
#     print(train,test)

#留p法
# from sklearn.model_selection import LeavePOut
# lpo=LeavePOut(p=2)

#自助法
# from sklearn.utils import resample
# y=df["GH(eV)"]#加了values会成为二维数组
# X=df.drop(["atom","GH(eV)","Size"],axis=1)
# X_train,y_train=resample(X,y)
# index_test=set(X.index)-set(X_train.index)#采用集合去重，方便相减
# X_test=X.iloc[list(index_test)]#iloc提取
# y_test=y[list(index_test)]
# X_train.shape,X_test.shape,y_test.shape,y_train.shape

# lr_bs=LinearRegression()
# lr_bs.fit(X_train,y_train)
# RMSE=np.sqrt(mean_squared_error(y_test,lr_bs.predict(X_test)))
# print(RMSE)