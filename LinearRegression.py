import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

cells=pd.read_csv('cells.csv')

x_train=cells[['time']]
y_train=cells[['cells']]

#create instance of model

reg=linear_model.LinearRegression()

#train model 

reg.fit(x_train, y_train)

#print("Predicted # cells : ",reg.predict([[2.3]]))
#print(reg.coef_)
#print(reg.intercept_)

cells_predict_df=pd.read_csv('cells_predict.csv')
predicted_cells=reg.predict(cells_predict_df)

cells_predict_df['cells']=predicted_cells

cells_predict_df.to_csv('test.csv')

cel=pd.read_csv('test.csv')
print(cel.head())
