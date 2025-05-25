#M.Rupp,A.Tkatchenko,et.al.Phys.Rev.Lett,2012,108,058301
#库伦矩阵构建分子/晶体描述符

#预测钙钛矿形成能
from matminer.datasets import load_dataset
df=load_dataset("文件名",data_home="文件当前文件夹")
18000*10

df=df[["formule","structure","e_form"]]
#前面两列构建描述符
#最后一列用于预测

from matminer.featurizers.structure.matrix import CoulombMatrix,SineCoulombMatrix#晶体的周期性适合于正弦库伦矩阵
import pandas as pd

s=df["stuctures"][0]

#构建计算器
cm_test=CoulombMatrix(flatten=False)#不压缩成一维
temp1=pd.DataFrame(cm_test.featurize(s)[0])#二维数组取第一个矩阵，5*5，5对应5个元素
#转化为二维表格输出

scm_test=SineCoulombMatrix(flatten=False)
temp2=pd.DataFrame(cm_test.featurize(s)[0])

#对角值一致，非对角的实对称
#0.5293采用的是bohn

cm=CoulombMatrix()#压成一维
df1=cm.fit_featurize_dataframe(df,"stucture")

scm=SineCoulombMatrix()#压成一维
df2=scm.fit_featurize_dataframe(df,"stucture")

X=df2.drop(["formule","structure","e_form"],axis=1).values
y=df2["e_form"].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
#默认百分之二十五

X_train.shape,X_test.shape,y_train.shape,y_test.shape

from xgboost import XGBRegressor
xgb=XGBRegressor()
xgb.fit(X_train,y_train)
xgb.score(X_train,y_train)#直接得出r2

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(y_train,xgb.predict(X_train),c="blue",s=1,alpha=0.5)
plt.scatter(y_test,xgb.predict(X_test),c="red",s=1,alpha=0.5)
plt.plot([-1,6],[-1,6],"k--")
plt.show()

#另一种描述符，描述原子元素的
#Magpie属性
#将(o,n,cl)转化为composition

from matminer.featurizers.conversions import StrToComposition
df3=StrToComposition().featurize_dataframe(df,"formula")#转换器，将化学式从字符串转化为某种数据结构

from matminer.featurizers.composition import ElementProperty
ep_feat=ElementProperty.from_preset(preset_name="magpie")

df3=ep_feat.featurize_dataframe(df3,"composition")
#多出来132列元素信息
#与晶体结构无关

X2=df3.drop(["formula","structure","e_form","composition"],axis=1).values

#交叉验证
from sklearn.model_selection import cross_val_predict
y_cv2=cross_val_predict(xgb,X2,y,cv=5)
#结果非常糟糕r2=0.005

#改良
#添加氧化态的信息
from matminer.featurizers.conversions import CompositionToOxidComposition
df3=CompositionToOxidComposition().featurize_dataframe(df3,"composition")
from matminer.featurizers.composition import OxidationStates
os_feat=OxidationStates()
df3=os_feat.featurize_dataframe(df3,"composition_oxid")
#增加了3列

#r2=0.055还是不好

#再增加上晶体结构
from matminer.featurizers.structure import DensityFeatures#密度/拥挤程度
df_feat=DensityFeatures()
df3=df_feat.featurize_dataframe(df3,"structure")
#增加了3列

#r2=0.76，rmse=0.364(0.721)

#数据/描述符决定了机器学习的上限
#模型/算法只是逼近这个上限

#4类常用描述符
#1.元素/原子序数
#2.晶体结构方面
#3.电子性质(d带中心)
#4.能量，吸附能