from sklearn import tree

X=[[1,1],[1,2],[2,1],[2,2]]
y=[0,1,1,1]
clf=tree.DecisionTreeClassifier()
clf.fit(X,y)

# a=clf.predict([[3,3]])
# print(a)

#概率预测
# b=clf.predict_proba([[1.2,1.4]])
# print(b)

#简单的画图
# from sklearn.tree import export_text
# tr=export_text(clf)
# print(tr)

#流程图可视化
#graphviz相关

X=[[1,1],[1,2],[1,3],[2,2],[2,1],[3,1]]
y=[1,1,1,0,1,0]
clf=tree.DecisionTreeClassifier(criterion="entropy")#默认为gini
clf.fit(X,y)
#sklearn并不配备C4.5算法

#简单的画图
from sklearn.tree import export_text
tr=export_text(clf)
print(tr)