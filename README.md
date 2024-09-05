# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GOGINENI BIDHISHA
RegisterNumber:212223040048
 
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df)
print()
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
## DATA SET
![image](https://github.com/user-attachments/assets/38250ae5-f115-4097-b5ba-618bb5dd0735)
## HEAD VALUES
![image](https://github.com/user-attachments/assets/ad08fbd1-0983-4b8a-acc7-8d94b5f231dc)
## TAIL VALUES
![image](https://github.com/user-attachments/assets/c917a99f-a8fb-40ca-abff-70d3ba123824)
## X and Y VALUES
![image](https://github.com/user-attachments/assets/33cf812f-00ea-42af-a0f9-ea1d61db302a)
## Predication values of X and Y
![image](https://github.com/user-attachments/assets/6a872593-3f8e-4ca9-abc9-8837a1ec6e79)

## GRAPH
![image](https://github.com/user-attachments/assets/4ad2eeed-e5f6-4c08-94df-ba0467e4c78e)
![image](https://github.com/user-attachments/assets/7b4bf303-40a2-43a5-82c9-8b815d9763ce)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
