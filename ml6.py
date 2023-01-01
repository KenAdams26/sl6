#6. Write a python program to implement POLYNOMIAL LINEAR REGRESSION for given dataset.
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  
# Importing the dataset
datas = pd.read_csv('city_temperature.csv')
datas

X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'blue')

plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.show()


