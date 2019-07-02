import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
house = load_boston()
bos = pd.DataFrame(house.data)
bos.columns = house.feature_names
bos['Price']=house.target
print(bos.head())
bos.describe()
import seaborn as sns
correl = bos.corr()
print(correl['Price'].sort_values(ascending=False)[:6], '\n')
print(correl['Price'].sort_values(ascending=False)[-6:])

quality_pivot = bos.pivot_table(index='CRIM', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

quality_pivot = bos.pivot_table(index='INDUS', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

quality_pivot = bos.pivot_table(index='NOX', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

quality_pivot = bos.pivot_table(index='AGE', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

quality_pivot = bos.pivot_table(index='CRIM', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

quality_pivot = bos.pivot_table(index='RAD', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

quality_pivot = bos.pivot_table(index='TAX', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

quality_pivot = bos.pivot_table(index='PTRATIO', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

quality_pivot = bos.pivot_table(index='LSTAT', values='Price', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.show()

X = bos.drop('Price',axis = 1)
Y = bos.Price
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 2)


from sklearn import linear_model
lr1 = linear_model.LinearRegression()
model = lr1.fit(x_train, y_train)

print('r2 is: ', model.score(x_test, y_test))
prediction = model.predict(x_test)
from sklearn.metrics import mean_squared_error
print('rmse: ', mean_squared_error(y_test, prediction))
