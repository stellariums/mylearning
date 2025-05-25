#支持向量机回归
import pandas as pd
df_train=pd.read_csv("OER(training).csv")
df_test=pd.read_csv("OER(predict).csv")
df_res=pd.read_csv("OER(target).csv")

X=df_train.drop(["Element","First"],axis=1)
y=df_res["X(DFT)"].values
X.shape,y.shape

#因为支持向量机涉及距离，需要归一化或者标准化，这里使用归一化
from sklearn.preprocessing import MinMaxScaler
#构建归一化器
mms=MinMaxScaler()
X_mms=mms.fit_transform(X)

#R代表回归/SVC，C代表分类
from sklearn.svm import SVR
svr=SVR()

#交叉验证
import numpy as np
from sklearn.model_selection import cross_val_score
#交叉验证无需fit
rmse_scores=cross_val_score(svr,X_mms,y,scoring="neg_root_mean_squared_error",cv=15)
np.mean(rmse_scores)

#预测
svr.fit(X_mms,y)
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(y,svr.predict(X_mms),s=80,c="cyan",edgecolors="k")
plt.xlim([-1.8,-0.6],[-1.8,-0.6],"r--")
plt.ylim([-1.8,-0.6],[-1.8,-0.6],"r--")
plt.plot([-1.8,-0.6],[-1.8,-0.6],"r--")
plt.show()

#皮尔逊系数
df=X
df["eta"]=y#合并XY
#分布情况
df.describe()

plt.scatter(df["radius[pm]"],df["eta"])#单独从某一指标比较
plt.scatter(df["d electron"],df["eta"])

import plotly.express as px#静态三维图
fig=px.scatter_3d(df,x="radius[pm]",y="d electron",z="eta",color="eta")#根据eta的大小来涂色
fig.show()

#衡量相关大小
#使用pearson系数
df.corr()#输出pearson系数的对称矩阵
df.corr()["eta"]#与Y相关

#可视化
import seaborn as sns
plt.figure(figsize=(5,1))
sns.heatmap(df.corr()["eta"].to_frame.T,cmap="RdYlGn",annot=True)
#热力图显示，转二维矩阵，转置，呈现对应的Pearson系数
plt.show()

#对比所有的XY，可以发现重叠程度大的描述符
plt.figure(figsize=(5,5))
sns.heatmap(df.corr(),cmap="RdYlGn",annot=True)
plt.show()