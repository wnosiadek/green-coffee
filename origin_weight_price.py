# classify origins based on price and weight
# use k-nearest neighbors classifier

import pandas
import origins
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier

# read relevant training data from the adjusted file
data = pandas.read_csv('adjusted_data.csv')
train_features = data[['Weight', 'Price']]
train_origins = data['Coffee']

# read relevant testing data from the adjusted file
test_data = pandas.read_csv('adjusted_test_data.csv')
test_features = test_data[['Weight', 'Price']]
test_origins = test_data['Coffee']

# transform training and testing data using origins.py
# (simplify possible origins,
#  take into account continents rather than countries,
#  change alphabetical data into numerical data)
train_features, train_origins = origins.simplify_origins(train_features, train_origins)
test_features, test_origins = origins.simplify_origins(test_features, test_origins)
# print the mapping for reference
print()
print(origins.printable_origins_map)

# plot the data
pyplot.scatter('Weight', 'Price', data=train_features, c=train_origins, cmap='gist_rainbow')
pyplot.colorbar(ticks=train_origins)
pyplot.xlabel('Weight kg')
pyplot.ylabel('Price PLN/kg')
pyplot.title('Green coffee origin classification')
pyplot.scatter('Weight', 'Price', data=test_features, c='black')
pyplot.show()

# convert the data to numpy arrays
train_features = train_features.to_numpy()
train_origins = train_origins.to_numpy()
test_features = test_features.to_numpy()
test_origins = test_origins.to_numpy()

# create a k-nearest neighbors classifier for the training data
classifier = KNeighborsClassifier()
classifier.fit(train_features, train_origins)

# make predictions on the testing data
predicted_origins = classifier.predict(test_features)
accuracy = classifier.score(test_features, test_origins)

# the results
print()
print(f'True origins: {test_origins}')
print(f'Predicted origins: {predicted_origins}')
print(f'Accuracy: {accuracy:.2f}')
