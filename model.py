import pandas as pd
df= pd.read_csv("adult.csv")
import numpy as np
df = df.fillna(np.nan)
df['income']=df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
df1= df.replace(to_replace ="?", value =np.nan)
df2= df1.dropna()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df2['workclass']= le.fit_transform(df2['workclass'])
df2['race']= le.fit_transform(df2['race'])
df2['relationship']= le.fit_transform(df2['relationship'])
df2['education']= le.fit_transform(df2['education'])
df2['native.country']= le.fit_transform(df2['native.country'])
df2['occupation']= le.fit_transform(df2['occupation'])
# Convert Sex value to 0 and 1
df2["sex"] = df2["sex"].map({"Male": 0, "Female":1})

# Create Married Column - Binary Yes(1) or No(0)
df2["marital.status"] = df2["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
df2["marital.status"] = df2["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
df2["marital.status"] = df2["marital.status"].map({"Married":1, "Single":0})
df2["marital.status"] = df2["marital.status"].astype(int)
t= df2.iloc[:,:-1]
r= df2.iloc[:,-1]

from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier(n_estimators=250,max_features=5)
rfc1.fit(t, r)

import pickle
pickle.dump(rfc1, open('mark4.pkl','wb'))