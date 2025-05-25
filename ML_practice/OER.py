import pandas as pd

df1=pd.read_csv("OER_train.csv")
X=df1.drop(["name","O","OH"],axis=1)
y=df1["O"].values
X.shape,y.shape

from sklearn.utils import shuffle

X_r,y_r=shuffle(X,y)#打乱顺序
X_train,y_train=X_r[:-100],y_r[:-100]
X_test,y_test=X_r[-100:],y_r[-100:]
X_test.shape,X_train.shape,y_test.shape,y_train.shape

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)

#交叉验证
from sklearn.model_selection import cross_val_score
import numpy as np#r2为数组
r2_scores=cross_val_score(lr,X,y,scoring="r2",cv=5)#默认折数为5
#help(cross_val_score)查看参数
np.mean(r2_scores)

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(y_train,lr.predict(X_train))
plt.scatter(y_test,lr.predict(X_test))
plt.plot([-4,8],[-4,8],"k--")
plt.xlim(-4,8)
plt.ylim(-4,8)
plt.show
#lr.coef_!=0全为true

#岭回归
from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(X_train,y_train)
#alpha默认为1，后续可调

#Lasso
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(X_train,y_train)
np.sum(lasso.coef_!=0)

lasso2=Lasso(alpha=0.1,max_iter=10000)#最大回归次数

#弹性网
from sklearn.linear_model import ElasticNet
enet=ElasticNet()
#help(ElasticNet)有alpha和l两个参数
y_pred_enet=enet.fit(X_train,y_train).predict(X_test)
np.sum(enet.coef_!=0)

#下面对alpha进行调参
import numpy as np
array_alpha=np.logspace(-2,1)#一系列0.01到10的等比数列
rmse=[]
for alpha in array_alpha:
    ridge=Ridge(alpha=alpha)
    rmse_scores=-cross_val_score(ridge,X,y,scoring="neg_root_mean_squared_error")
    rmse.append(np.mean(rmse_scores))

plt.plot(array_alpha,rmse)
plt.xscale("log")
plt.show()