import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

cells=pd.read_csv('data/cells.csv')

x_train=cells[['time']]
y_train=cells[['cells']]

print(x_train)
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.4,random_state=10)

#create instance of model

reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
predication_test=reg.predict(x_test)
print(predication_test)
print(y_test)
print("Mean squre error mean = ",np.mean(predication_test - y_test)**2)

plt.scatter(predication_test,predication_test - y_test)
plt.hlines(y=0, xmin=200, xmax=310)

