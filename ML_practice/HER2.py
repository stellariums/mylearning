#神经网络算法
import pandas as pd
df=pd.read_csv("HER2.csv",index_col=0)#列索引
X=df.drop("Eads(H)",axis=1)
y=df["Eads(H)"]
X.shape,y.shape

#直接切割
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)#测试集比例
X_train.shape,X_test.shape,y_train.shape,y_test.shape

#归一化
X_train_s=(X_train-X_train.min())/(X_train.max()-X_train.min())
X_test_s=(X_test-X_train.min())/(X_train.max()-X_train.min())

from sklearn.neural_network import MLPRegressor
nn=MLPRegressor(hidden_layer_sizes=(64,64))#两层
nn.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error as mse,r2_score
MSE=mse(y_train,nn.predict(X_train))

from matplotlib.pyplot import plt
