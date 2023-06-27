"""
Script for classifying green coffee origin based on its price and weight

Transforms the data using 'origins.py', visualizes the data, creates a Decision Tree Classifier and fits it on the
training set, visualizes the tree, tests the model on the testing set

Requires installation of 'pandas', 'matplotlib', 'scikit-learn'
"""

import data
import origins
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier, plot_tree

print('\nClassifying green coffee origin based on its price and weight...')

# get relevant training and testing data using 'data.py'
train_features, train_origins, test_features, test_origins = data.get_data(['Weight', 'Price'], 'Coffee')

# transform training and testing data using 'origins.py'
# (simplify possible origins, take into account continents rather than countries, change alphabetical data into
#  numerical data)
train_features, train_origins = origins.simplify_origins(train_features, train_origins)
test_features, test_origins = origins.simplify_origins(test_features, test_origins)
# print the mapping for reference
print(f'\n{origins.printable_origins_map}')

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

# create a Decision Tree Classifier for the training data
classifier = DecisionTreeClassifier()
classifier.fit(train_features, train_origins)

# plot the decision tree
plot_tree(classifier, max_depth=2, feature_names=['Weight', 'Price'], class_names = ['0', '1', '2', '3'])
pyplot.title('Green coffee origin classification - decision tree')
pyplot.show()

# make predictions on the testing data
predicted_origins = classifier.predict(test_features)
accuracy = classifier.score(test_features, test_origins)

# the results
print(f'\nTrue origins: {test_origins}')
print(f'Origins predicted with Decision Trees: {predicted_origins}')
print(f'Accuracy (mean accuracy): {accuracy:.2f}')
