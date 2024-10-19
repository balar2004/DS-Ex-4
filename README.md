## Ex.No:4 Feature Scaling and Selection
## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

## FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

## FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method.

2.Wrapper Method.

3.Embedded Method.

## CODING AND OUTPUT:
## FEATURE SCALING:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/591f56f7-4377-4b94-8e37-bffe510091b5)
```
df.dropna()
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/9d1400a8-b89f-4087-8628-7a8d7ecbf488)
```
max_vals=np.max(np.abs(df[['Height']]))
max_vals1=np.max(np.abs(df[['Weight']]))
print("Height =",max_vals)
print("Weight =",max_vals1)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/a50d62e2-6f3a-4503-a0cf-ec770c9cfff6)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/11c95171-e62e-4aaf-9c4c-7a1c3d9b9179)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/b2842846-a312-4304-80a5-ce982fc031d3)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/0255d514-bc07-431d-9f91-9b46d9157610)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/5ed0927e-525b-47b1-b4ec-45c550124312)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/e6bf26b9-a791-47dd-b76e-b93c4674dd05)

## FEATURE SELECTION:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/6b69f437-999d-41cd-82e1-42bcb1852d5e)

```
data.isnull().sum()
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/cc4a7852-2f5a-4cda-a3e5-00f3bd3adb4c)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/d7f20ffd-8ed6-46c9-827e-002734d7c24e)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/606f3c60-1b52-4e9c-aedb-bcc75beb9c20)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/d48a4d07-4e07-4d55-9cf5-38b89a928c7d)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/3786e648-d190-4dec-825d-40be22542b71)
```
data2
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/8f71e0aa-65ad-492a-8e46-2b0336923780)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/0939b8d6-8cbb-4db1-baf3-d588ed201eb0)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/ebcf37da-b3cd-45c5-acbd-3ed4cb0877eb)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/98dac718-6dd8-4c25-9871-a4ac7fe7f53b)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/5fdf6350-f61e-4679-8b66-43fc938756c9)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/5e9e3c48-c448-4c5b-8f6a-9b036cf0aea4)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/786ea75d-e5b6-4a5d-81d7-522a4a5dab8f)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/863d2f0f-5fba-4a6a-a089-4aae04179cfb)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/41aa533b-8bc7-4380-92c0-7e30b56fdf29)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/872be69c-39dc-45df-ac0f-87862a3052bd)

```
data.shape
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/af135c3e-40a9-414c-869d-5951eaf21d12)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data= {
    'Feature1' : [1,2,3,4,5],
    'Feature2' : ['A','B','C','A','B'],
    'Feature3' : [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector = SelectKBest(score_func= mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/2c496798-718b-4e9c-8548-75ac82108d42)

## Contigency Table
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/05b28951-1472-4ac9-b882-6a508d1a3c2d)

```
contingency_table = pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/af7bd8e7-85f1-4122-b12b-bfc5ece25ad0)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/121243595/d5adf339-c98b-48f0-baf1-afb6f0ffffa2)


## RESULT:
Thus, Feature selection and Feature scaling has been used on the given dataset.
