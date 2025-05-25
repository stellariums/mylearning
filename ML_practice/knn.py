import pandas as pd
df=pd.read_csv("clfz.csv")

X0,X1=df['x'],df['y']
X=df[['x','y']].values
y=df['label']
X.shape,y.shape

from sklearn.neighbors import KNeighborsClassifier
clf_knn=KNeighborsClassifier()
clf_knn.fit(X,y)

import numpy as np
xx,yy=np.meshgrid(np.arange(-45,45,0.5),np.arange(-45,45,0.5))
coords=np.stack((xx.reshape(-1),yy.reshape(-1)),axis=1)
Z=clf_knn.predict(coords)
Z=Z.reshape(xx.shape)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_res(X,y,xx,yy,Z):
    light_rgb=ListedColormap(["#AAAAFF","#FFAAAA"])
    plt.figure(figsize=(5,5))
    plt.pcolormesh(xx,yy,Z,cmap=light_rgb)
    plt.scatter(X0,X1,c=df['label'],cmap="bwr",s=20)
    plt.show()

#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaler=scaler.fit_transform(X)
clf_knn=KNeighborsClassifier()
clf_knn.fit(X_scaler,y)

xx_s,yy_s=np.meshgrid(np.arange(-45,45,0.5),np.arange(-45,45,0.5))
coords_s=np.stack((xx_s.reshape(-1),yy_s.reshape(-1)),axis=1)
coords_s=scaler.transform(coords_s)
Z_s=clf_knn.predict(coords_s)
Z_s=Z_s.reshape(xx_s.shape)#存疑，是否要逆变换
plot_res(X0,X1,xx_s,yy_s,Z_s)