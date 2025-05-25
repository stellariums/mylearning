import numpy as np
import pandas as pd

columns_names=["formula","db_center","ad_energy"]
data=pd.read_csv("C:/Users/20414/Desktop/data/d_bond_center_data.csv",names=columns_names)#数据文件

x=data["db_center"].values.reshape(-1,1)#此处需要二维数组
y=data["ad_energy"].values#导入数据

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)#拟合过程

coef=lr.coef_
intercept=lr.intercept_

predicts=lr.predict([[-2.2],[-2.3]])#预测

#R2=lr.score(x,y)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
r2=r2_score(y_true=y,y_pred=lr.predict(x))
mae=mean_absolute_error(y_true=y,y_pred=lr.predict(x))
mse=mean_squared_error(y_true=y,y_pred=lr.predict(x))
#指标

#绘图
import matplotlib.pyplot as plt
x_=x.reshape(-1)
plt.scatter(x_,y,label="actual_value")
plt.plot(x_,lr.predict(x),color="red",label="predicted_value")
plt.legend()
plt.show()