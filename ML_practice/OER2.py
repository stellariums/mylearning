import pandas as pd
df=pd.read_csv("C:/Users/20414/Desktop/data/clf1.csv")
#print(df)
X=df[["O-2p band center(eV)","(O-2pM-d)(eV)"]].values
y=df["Reaction mechanism"].values

from sklearn.tree import DecisionTreeClassifier
clf_tree=DecisionTreeClassifier()
clf_tree.fit(X,y)

#可视化
import matplotlib.pyplot as plt
# plt.figure(figsize=(5,5))#尺寸
# plt.scatter(X[:,0],X[:,1],c=y=="LOM",cmap="RdYlGn",s=100)
# plt.xlim(-4.0,0.0)
# plt.ylim(-3.0,1.0)
# plt.show()

#分离器的可视化
import numpy as np
#生成网格点
xx,yy=np.meshgrid(np.arange(-4.0,0.0,0.01),np.arange(-3.0,1.0,0.01))
#拼接
coords=np.stack((xx.reshape(-1),yy.reshape(-1)),axis=1)#按列
Z=clf_tree.predict(coords)

#画入图中
Z=Z.reshape(xx.shape)#转二维
#找颜色
from matplotlib.colors import ListedColormap
def plot_res(xx,yy,Z):
    light_rgb=ListedColormap(["#FFAAAA","#AAFFAA"])
    plt.figure(figsize=(5,5))
    plt.pcolormesh(xx,yy,Z=="LOM",cmap=light_rgb)
    plt.scatter(X[:,0],X[:,1],c=y=="LOM",cmap="RdYlGn",s=100)
    plt.xlim(-4.0,0.0)
    plt.ylim(-3.0,1.0)
    plt.show()
# plot_res(xx,yy,Z)

#若使用ID3算法
#clf_tree=DecisionTreeClassifier(criterion="entropy")

#限定深度，获取泛化性更强的模型
#clf_tree=DecisionTreeClassifier(max_depth=2)

#逻辑回归
from sklearn.linear_model import LogisticRegression#广义的线性回归
clf_lr=LogisticRegression()
clf_lr.fit(X,y)
# Z=clf_lr.predict(coords)
# Z=Z.reshape(xx.shape)
# plot_res(xx,yy,Z)

#效果评估
#我们使用准确度
from sklearn.metrics import accuracy_score
acc=accuracy_score(y,clf_tree.predict(X))
#print(acc)

#留出法
from sklearn.utils import shuffle
#打乱
X_r,y_r=shuffle(X,y)
X_train,y_train=X_r[:-10],y_r[:-10]
X_test,y_test=X_r[-10:],y_r[-10:]
X_train.shape,X_test.shape

clf_tree.fit(X_train,y_train)
acc1=accuracy_score(y_test,clf_tree.predict(X_test))
# print(acc1)
clf_tree2=DecisionTreeClassifier(criterion="entropy")
clf_tree2.fit(X_train,y_train)
acc2=accuracy_score(y_test,clf_tree2.predict(X_test))
# print(acc2)
clf_lr.fit(X_train,y_train)
acc3=accuracy_score(y_test,clf_lr.predict(X_test))
# print(acc3)
clf_tree3=DecisionTreeClassifier(max_depth=2)
clf_tree3.fit(X_train,y_train)
acc2=accuracy_score(y_test,clf_tree3.predict(X_test))
# print(acc2)


#交叉验证法
from sklearn.model_selection import cross_val_score
acc1=cross_val_score(clf_tree,X,y,cv=5)#默认指标为accuracy
print(np.mean(acc1))
acc2=cross_val_score(clf_tree2,X,y,cv=5)#默认指标为accuracy
print(np.mean(acc2))
acc3=cross_val_score(clf_tree3,X,y,cv=5)#默认指标为accuracy
print(np.mean(acc3))
acc4=cross_val_score(clf_lr,X,y,cv=5)#默认指标为accuracy
print(np.mean(acc4))