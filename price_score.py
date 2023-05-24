"""
Script for investigating the dependence of green coffee price on its score

Creates a Linear Regression model for the data, visualizes the data and the results

Requires installation of 'pandas', 'matplotlib', 'scikit-learn'
"""

import pandas
import adjust_data
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression

# read relevant data from the adjusted file
data = pandas.read_csv('adjusted_data.csv')
score = data['Score']
price = data['Price']
# drop rows with missing values using adjust_data.py
score, price = adjust_data.drop_missing(score, price)

# plot the data
pyplot.scatter(score, price)
pyplot.xlabel('SCA score')
pyplot.ylabel('Price PLN/kg')
pyplot.title('Green coffee price against its score')

# convert the data to numpy arrays
score = score.to_numpy().reshape(-1, 1)
price = price.to_numpy()

# create a linear regression model for the data
model = LinearRegression()
model.fit(score, price)
predicted_price = model.predict(score)

# details of the fitted model
a = model.coef_[0]
b = model.intercept_
r2 = model.score(score, price)
print(f'y = {a:.0f}x + {b:.0f}\nR2 = {r2:.2f}')
# y = 6x + -459
# R2 = 0.48

# plot the fitted model and its details
pyplot.plot(score, predicted_price, color='black')
pyplot.text(87.5, 90, f'y = {a:.0f}x + {b:.0f}\nR2 = {r2:.2f}')
pyplot.show()
