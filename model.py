import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

"""# Data collection  & processing"""

#loading the csv data to pandas dataFrame
gold_data=pd.read_csv('gld_price_data.csv')

#print first 5 rows in dataFrame
gold_data.head()

#print last 5 rows in dataFrame
gold_data.tail()

#numbers_of_rows_and_columns
gold_data.shape

#getting basic info about data
gold_data.info()

#checking numbers of missing values
gold_data.isnull().sum()

#getting statistical measures of the data
gold_data.describe()

"""correlation:

1-positive correlation

2-negative correlation
"""

correlation=gold_data.drop('Date',axis=1).corr()
#constructing heatmap to understand correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()
#

# correlation values of gold
print(correlation['GLD'])

#checking distrubtion of the GLD price
sns.distplot(gold_data['GLD'],color='green')

"""splitting features & target"""

x=gold_data.drop(['Date','GLD'],axis=1)
y=gold_data['GLD']

print(x)

print(y)

"""Splitting to train data & test data"""

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

"""Model Training:

  Random Forest Regressor

"""

regressor=RandomForestRegressor(n_estimators=100)

#training model
regressor.fit(x_train,y_train)

"""# Model Evalution (make some predection)"""

# predection on  test data
test_data_prediction = regressor.predict(x_test)

print(test_data_prediction)

from logging import error
# R squared error
error_score=metrics.r2_score(y_test,test_data_prediction)
print("R squared error : ",error_score)
#

"""compare actual and predicted values in plot"""

y_test=list(y_test)
plt.plot(y_test,color='blue',label='Actual Value')
plt.plot(test_data_prediction,color='green',label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
#
