# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values. 

## Programs :

```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Rakshitha P
RegisterNumber:  212223220083
```

```python
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
```
## Output :
![image](https://github.com/SANTHAN-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/80164014/cd5ed94d-ff93-4bb2-9f74-3f6bc161875e)

```python
data.isnull().sum()
```
## Output :
![image](https://github.com/SANTHAN-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/80164014/73fda4c5-f77a-4529-b036-76a6d587c992)


```python
data["left"].value_counts
```
## Output :
![image](https://github.com/SANTHAN-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/80164014/b56af4d5-8747-4288-8e44-6b6dada52fe2)



```python
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
## Output :
![image](https://github.com/SANTHAN-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/80164014/45a2811a-1f93-4184-9a06-7f5ce26712a5)


```python
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
```
## Output :
![image](https://github.com/SANTHAN-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/80164014/39075de0-7bc5-43ee-a51a-204a50962a27)


```python
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
```

## Output :
![image](https://github.com/SANTHAN-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/80164014/c46d09ed-3ab8-4af9-b12d-bcd9ec52fed5)


```python
y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output :
![image](https://github.com/SANTHAN-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/80164014/d042abfa-3736-4741-be7a-40a17e5a6d31)


```python
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output :
![image](https://github.com/SANTHAN-2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/80164014/5bd862e8-6586-4e14-8ac0-2b88aa02b579)

## Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
