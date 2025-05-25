#预处理
#cpd.chemcatbio.org

import json
import pandas as pd

with open("C:/Users/20414/Desktop/cpd.json",encoding="UTF8") as json_file:
    data=json.load(json_file)

# print(data.keys())#两个key
# print(len(data["results"]))
#print(data["results"][0])

# 直接遍历字典列表，无需额外嵌套
df = pd.DataFrame([data_single["adsorption_measurement"] for data_single in data["results"]])
#print(df)
#df=pd.DataFrame(df[0])


# 显示列索引值
#print(df.columns)

#提取衬底物种
df["substrate"]=df['adsorbate_fraction'].apply(lambda x:x.get("unit",None))#应用get,若提取不出，返回None
#print(df)

#晶面信息
df["dict_facet"]=df["bulk_surface_property_set"].apply(lambda x:x.get("facet",None))

#吸附的物种
df["adsorbate"]=df["adsorbate_species"].apply(lambda x:x.get("formula",None))

#参考系数
#df["adsorption_reference_species_set"])#实为列表，需要转变为字典

df["reference_coefficient"]=df["adsorption_reference_species_set"].apply(lambda x:x[0].get("reference_coefficient",None))
df=df[["substrate",'dict_facet','adsorbate','reference_coefficient','adsorption_site','adsorption_energy']]
#注意要加两个括号
#print(df)

#统计无效数据
#print(df.isna().sum())

#丢掉无效数据
df=df.dropna(subset=["substrate","dict_facet","adsorption_site"])
#编号不连续了
#print(df)

#重排序号
df=df.reset_index(drop=True)
#不添加新的列
#print(df)

#print((set(df["substrate"])))#查看衬底信息
#只要金属单晶的
df=df[~df["substrate"].isin(['PdO','Cu+Pd','surface atom','Pt+Fe','Pt+Mo'])]
df=df.reset_index(drop=True)
#print(df)

#晶面处理
df["facet"]=df["dict_facet"].apply(lambda x:x.get("name",None))
df=df.drop("dict_facet",axis=1)
#print(df)

#只考虑单原子吸附
#print((set(df["adsorbate"])))
#选取长度为一，但是Cl等单原子无法选择
df=df[df["adsorbate"].str.len()<=1]
df=df.reset_index(drop=True)
#print(df)

#处理参考系数0.5/1.0
#print((set(df["reference_coefficient"])))
df=df[df["reference_coefficient"]==1]#别忘了==
df=df.reset_index(drop=True)
#print(df)

#处理吸附位点
#df["adsorption_site"]=df["adsorption_site"].apply()
#问题是位点的表示乱七八糟，得一个一个输入
adsorption_site_name=df["adsorption_site"].apply(lambda x: x.get('site_name'))
#print(set(adsorption_site_name))

#转换为周边原子个数
site2num={
    'H123':3,
    'top-down':1,
    'f':3,
    'three-fold hollow':4,
    'step hollow':4,
    'B23':2, 
    'b8':2, 
    'al3':3,
    'hcp':3, 
    'fourfold hollow':4, 
    'h':3, 
    'h3':3, 
    'f0':3, 
    't4':4, 
    'b2':2, 
    'Top':1, 
    'top/fcc':3, 
    't3':3, 
    'h2':3, 
    'Br':1, 
    'hcp-Pt':3, 
    'long bridge':2, 
    'Pd3':3,
    'b3':3, 
    'hollow, η3':3, 
    'top':1, 
    'lb':2, 
    'H_C(l)':3, 
    'step bridge':2, 
    'bridge':2, 
    'bridge-top':2, 
    'h5':3, 
    'b1':2, 
    'pseudo-three-fold':3, 
    'b':2, 
    't':3, 
    'ti1':3, 
    '4f':4, 
    '3-fold hollow':4, 
    'fcc':3, 
    'hcp-top':3, 
    'hcp II':3, 
    'hollow':4, 
    'Fcc/Hcp':3, 
    'three-fold hollow (F)':4,
    'b6':3, 
    'fcc II':3, 
    '3F':3, 
    'fcc-Mo':3, 
    'h1':3, 
    'fcc-Pt':3, 
    'hollow-u2dl':4, 
    't2':3, 
    'Surface fcc':3, 
    'hollow, threefold coordinated':4, 
    't1':3, 
    '4F':4, 
    '3f':3, 
    'Ag3':3, 
    'al1':1, 
    'pseudo four-fold hollow':4, 
    'Fcc':3
}#标签：周边原子个数（随便写的，不准确）

#推导式
df["site"]=[site2num[x] for x in adsorption_site_name]
df=df.drop("adsorption_site",axis=1)

#查重和去重
df=df.drop_duplicates()#根据列标签完全一致
df=df.reset_index(drop=True)
#print(df)

#但是前面几项相等，吸附能略有差别的会存在
#而且吸附能数值不一定为浮点数，可能为字符
#先转换
df["adsorption_energy"]=df["adsorption_energy"].astype(float)
df=df.groupby(["substrate","adsorbate","facet","site"]).agg({"adsorption_energy":"mean"}).reset_index()
#print(df)

#将信息转换为数据
from datana.atom import Atom
#衬底转化为原子的信息
df["atom_number"]=[Atom(x).get_number() for x in df['substrate']]#原子序数
#相对原子质量
df["atom_weight"]=[Atom(x).get_weight() for x in df['substrate']]
#get_periodic(),get_group()周期数和主族

#吸附物种
df["adsorbate_number"]=[Atom(x).get_number() for x in df["adsorbate"]]
df["adsorbate_weight"]=[Atom(x).get_weight() for x in df["adsorbate"]]

#晶面
#print(str(set(df['facet'])))
#print(df[df["facet"]=="(211)"])#数据很少，不平衡
df=df[~df["facet"].isin(["(11-21)",'(11-20)','(10-10)'])]
df=df.reset_index(drop=True)
#print(df)
#one-hot编码处理晶面信息
#晶面信息为并列，不能使用123，应该用01
df_oh=pd.get_dummies(df["facet"])
#拼接
df[df_oh.columns]=df_oh#获取列标签
#true and false不影响0和1

X=df.drop(["substrate","adsorbate","facet","adsorption_energy"],axis=1).values
y=df['adsorption_energy'].values
X.shape,y.shape
#print(X,y)

from xgboost import XGBRegressor
xgb=XGBRegressor()
#泛化性能强
from sklearn.model_selection import cross_val_predict,cross_val_score
import numpy as np
# rmse_scores=-cross_val_score(xgb,X,y,scoring="neg_root_mean_squared_error",cv=5)#注意负号
# r2_scores=cross_val_score(xgb,X,y,scoring="r2",cv=5)
#print(np.mean(rmse_scores),np.mean(r2_scores))
#不是很好，有待调参

# y_cv=cross_val_predict(xgb,X,y,cv=5)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(4,4))
# plt.scatter(y,y_cv)
# plt.plot([-10,1],[-10,1],"k--")
# plt.xlim(-10,1)
# plt.ylim(-10,1)
# plt.show()
from sklearn.model_selection import GridSearchCV
parameters={
    "n_estimators":list(range(50,650,50))#字典
}
gs=GridSearchCV(xgb,parameters,cv=20,scoring="neg_mean_squared_error")
gs.fit(X,y)
print(gs.best_params_)#输出最佳
print(gs.best_score_)#输出最佳分数
np.sqrt(-gs.best_score_)