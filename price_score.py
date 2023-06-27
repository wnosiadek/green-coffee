"""
Script for investigating the dependence of green coffee price on its score

Creates a Linear Regression model for the data, visualizes the data and the results

Requires installation of 'pandas', 'matplotlib', 'scikit-learn'
"""

import data
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression

print('\nInvestigating the dependence of green coffee price on its score...')

# get relevant training and testing data using 'data.py'
score, price = data.get_data('Score', 'Price', test=False)

# plot the data
pyplot.scatter(score, price)
pyplot.xlabel('SCA score')
pyplot.ylabel('Price PLN/kg')
pyplot.title('Green coffee price against its score')

# convert the data to numpy arrays
score = score.to_numpy().reshape(-1, 1)
price = price.to_numpy()

# create a Linear Regression model for the data
model = LinearRegression()
model.fit(score, price)
predicted_price = model.predict(score)

# details of the fitted model
a = model.coef_[0]
b = model.intercept_
r2 = model.score(score, price)
print(f'\nModel obtained with Linear Regression: y = {a:.0f}x + {b:.0f}')
print(f'Accuracy (coefficient of determination): R2 = {r2:.2f}')

# plot the fitted model and its details
pyplot.plot(score, predicted_price, color='black')
pyplot.text(87.5, 90, f'y = {a:.0f}x + {b:.0f}\nR2 = {r2:.2f}')
pyplot.show()
