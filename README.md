# EX3 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Get the independent variable X and dependent variable Y.<p>
 2.Calculate the mean of the X -values and the mean of the Y -values.<p>
 3.Find the slope m of the line of best fit using the formula.

![image](https://github.com/user-attachments/assets/23c20f80-7b52-4f7a-a48c-a5732334feb0)

4.Compute the y -intercept of the line by using the formula:

![image](https://github.com/user-attachments/assets/d0e6c793-94bf-4f57-970c-a98b7ff85269)

5.Use the slope m and the y -intercept to form the equation of the line.<p> 
6. Obtain the straight line equation Y=mX+b and plot the scatterplot. 
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: K.Nishal
RegisterNumber: 2305001021 
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iterations=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    for _ in range(num_iterations):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1, 1)

        theta -= learning_rate * (1/len(X1))  * X.T.dot(errors)

    return theta

data = pd.read_csv('/content/ex.3 ml.csv',header=None)
X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1, 1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/9b01a79b-0b6a-48b3-8b31-9e6945d1033e)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
