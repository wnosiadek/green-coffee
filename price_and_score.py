# investigate the dependence of green coffee price on its score
# use linear regression

import pandas
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression

# read relevant data from the adjusted file
# drop rows with missing values
data = pandas.read_excel('adjusted_data.xlsx')
data.dropna(inplace=True, ignore_index=True)
score = data['Score']
price = data['Price']

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

# plot the fitted model and its details
pyplot.plot(score, predicted_price, color='black')
pyplot.text(87.5, 90, f'y = {a:.0f}x + {b:.0f}\nR2 = {r2:.2f}')
pyplot.show()
