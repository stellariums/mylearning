import pandas as pd
df=pd.read_csv('clf3.csv')
import numpy as np#计算最大最小，画图用
np.min(df['x']),np.max(['x'])
np.min(df['y']),np.max(['y'])

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))#尺寸
plt.scatter(X0,X1,c=df['label'],cmap="bwr",s=20)
plt.xlim(-85,95)
plt.ylim(-85,95)
plt.show()

X=df[["x","y"]].values
y=df["label"]
X.shape,y.shape

from sklearn.naive_bayes import GaussianNB#此为连续变量所使用，离散变量使用Multinomial Naive Bayes
clf_nb=GaussianNB()
clf_nb.fit(X,y)

xx,yy=np.meshgrid(np.arange(-85,95,0.5),np.arange(-85,95,0.5))
coords=np.stack((xx.reshape(-1),yy.reshape(-1)),axis=1)
Z=clf_nb.predict(coords)
Z=Z.reshape(xx.shape)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_res(X,y,xx,yy,Z):
    light_rgb=ListedColormap(["#AAAAFF","#FFAAAA"])
    plt.figure(figsize=(5,5))
    plt.pcolormesh(xx,yy,Z,cmap=light_rgb)
    plt.scatter(X0,X1,c=df['label'],cmap="bwr",s=20)
    plt.show()

#SVC算法
from sklearn.pipeline import make_pipeline#管道
from sklearn.svm import SVC#支持向量机
from sklearn.preprocessing import StandardScaler#标准化

clf_svm=make_pipeline(StandardScaler(),SVC(kernel="linear"))
clf_svm.fit(X,y)

Z=clf_svm.predict(coords)
Z=Z.reshape(xx.shape)
plot_res()

clf_svm=make_pipeline(StandardScaler(),SVC(kernel="rbf"))#高斯核，效果更好
clf_svm.fit(X,y)

#性能度量
#查准率
from sklearn.metrics import precision_score
y_true=[0,1,0,0,1,0]
y_pred=[0,1,1,0,1,1]
precision_score(y_true,y_pred)
precision_score(y_true,y_pred,average=None)#全部类别输出
precision_score(y_true,y_pred,average="macro")#宏查准率——关注少样本
precision_score(y_true,y_pred,average="micro")#微查准率——关注多样本
precision_score(y_true,y_pred,average="weighted")#权重

#查全率/召回率
from sklearn.metrics import recall_score
recall_score(y_true,y_pred,average=None)

#PR曲线（查全率和查准率在很大时负相关）
#查全率和查准率在几率（阈值）变化时候会变化
from sklearn.metrics import PrecisionRecallDisplay#画图PR
display=PrecisionRecallDisplay.from_estimator(clf_nb,X,y,name="nb")
display.ax_.set_title("PR_Curve")#标题
plt.show()

display=PrecisionRecallDisplay.from_estimator(clf_nb,X,y,name="svm")
display.ax_.set_title("PR_Curve")
plt.show()

y_score=clf_nb.predict_proba(X)[:,1]
display=PrecisionRecallDisplay.from_predictions(y,y_score,name="nb")
plt.show()

#支持向量机大多不直接支持几率
#转化
clf_svm3=make_pipeline(StandardScaler(),SVC(kernel="linear",probability=True))

clf_svm3.fit(X,y)
y_score=clf_svm3.predict_proba(X)[:,1]
display=PrecisionRecallDisplay.from_predictions(y,y_score,name="svm3")
plt.show()

#选择模型
#BEP平衡点方法
from sklearn.metrics import precision_recall_curve#计算PR曲线各点值
#会返回三个数值
precision_nb,recall_nb,thresholds_nb=precision_recall_curve(y,clf_nb.predict_proba(X)[:,1])#thresholds为阈值
#P=1时，R不一定为0
precision_svm3,recall_svm3,thresholds_svm3=precision_recall_curve(y,clf_svm3.predict_proba(X)[:,1])

clf_svm4=make_pipeline(StandardScaler(),SVC(kernel="rbf",probability=True))
clf_svm4.fit(X,y)
precision_svm4,recall_svm4,thresholds_svm4=precision_recall_curve(y,clf_svm4.predict_proba(X)[:,1])

#画图
fig,ax=plt.subplots(figsize=(4,4))
ax.step(recall_nb,precision_nb,label="bayes")
ax.step(recall_svm3,precision_svm3,label="svm_linear")
ax.step(recall_svm4,precision_svm4,label="svm_rbf")
ax.set_xlim(0.90,1.02)#很大时才显现出差别
ax.set_ylim(0.90,1.02)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)#去边框
fig.legend()#显示标签
ax.plot([0,1],[0,1],"k--")#画对角线
plt.show()

#rbf>linear>bayes

#F1度量
from sklearn.metrics import f1_score
f1_score(y,clf_nb.predict(X))#无需概率

#ROC曲线评估
from sklearn.metrics import roc_curve
fpr_nb,tpr_nb,thresholds_nb=roc_curve(y,clf_nb.predict_proba(X)[:,1],pos_label=1)
fpr_svm3,tpr_svm3,thresholds_svm3=roc_curve(y,clf_svm3.predict_proba(X)[:,1],pos_label=1)

plt.figure(figsize=(4,4))
plt.plot(fpr_nb,tpr_nb,label="nb")
plt.plot(fpr_svm3,tpr_svm3,label="svm3")
plt.legend()
plt.show()

#若看不出来，可以使用AUG法求围成的面积
from sklearn.metrics import auc
roc_auc=auc(fpr_nb,tpr_nb)
#最终得出0~1的值，选最大