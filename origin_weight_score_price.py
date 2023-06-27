"""
Script for classifying green coffee origin based on its price, score and weight

Transforms the data using 'origins.py', scales the data with Min-Max Scaler and reduces its dimensionality with
Principal Component Analysis, visualizes the data, creates a Decision Tree Classifier and fits it on the training set,
visualizes the tree, tests the model on the testing set

Requires installation of 'pandas', 'scikit-learn', 'numpy', 'matplotlib'
"""

import data
import origins
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier, plot_tree

print('\nClassifying green coffee origin based on its price, score and weight...')

# get relevant training and testing data using 'data.py'
train_features, train_origins, test_features, test_origins \
    = data.get_data(['Weight', 'Score', 'Price'], 'Coffee')

# transform training and testing data using 'origins.py'
# (simplify possible origins, take into account continents rather than countries, change alphabetical data into
#  numerical data)
train_features, train_origins = origins.simplify_origins(train_features, train_origins)
test_features, test_origins = origins.simplify_origins(test_features, test_origins)
# print the mapping for reference
print(f'\n{origins.printable_origins_map}')

# convert the data to numpy arrays
train_features = train_features.to_numpy()
train_origins = train_origins.to_numpy()
test_features = test_features.to_numpy()
test_origins = test_origins.to_numpy()

# create a Min-Max Scaler for the training data
# scale the training features
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
# scale the testing features
test_features = scaler.transform(test_features)

print('\nReducing the dimensionality...')

# create a Principal Component Analysis model for the training data
# reduce the dimensionality of the training features
pca = PCA(n_components=2)    # 3 components -> 2 components
train_features_pca = pca.fit_transform(train_features)
# reduce the dimensionality of the testing features
test_features_pca = pca.transform(test_features)

# details of the PCA model
pca_components = pca.components_
pca_variances = pca.explained_variance_ratio_
# print the details
# (first set float printing options for numpy arrays)
numpy.set_printoptions(precision=2, floatmode='maxprec_equal')
print(f'\nPrincipal components in [Weight, Score, Price] space:\n{pca_components}')
print(f'Variance explained by each component ({sum(pca_variances):.2f} in total):\n{pca_variances}')

# plot the data
pyplot.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=train_origins, cmap='gist_rainbow')
pyplot.colorbar(ticks=train_origins)
pyplot.xlabel(f'I PC: [Weight, Score, Price] = {pca_components[0]}')
pyplot.ylabel(f'II PC: [Weight, Score, Price] = {pca_components[1]}')
pyplot.title('Green coffee origin classification')
pyplot.scatter(test_features_pca[:, 0], test_features_pca[:, 1], c='black')
pyplot.show()

# create a Decision Tree Classifier for the training data
classifier_pca = DecisionTreeClassifier()
classifier_pca.fit(train_features_pca, train_origins)

# plot the decision tree
plot_tree(classifier_pca, max_depth=2, feature_names=[f'I PC', f'II PC'], class_names = ['0', '1', '2', '3'])
pyplot.title('Green coffee origin classification - decision tree')
pyplot.show()

# make predictions on the testing data
predicted_origins_pca = classifier_pca.predict(test_features_pca)
accuracy_pca = classifier_pca.score(test_features_pca, test_origins)

# the results
print('\nWith Principal Component Analysis (2D)')
print(f'True origins: {test_origins}')
print(f'Origins predicted with Decision Trees: {predicted_origins_pca}')
print(f'Accuracy (mean accuracy): {accuracy_pca:.2f}')

# the same but without PCA

# create a Decision Tree Classifier for the training data
classifier = DecisionTreeClassifier()
classifier.fit(train_features, train_origins)

# make predictions on the testing data
predicted_origins = classifier.predict(test_features)
accuracy = classifier.score(test_features, test_origins)

# the results
print('\nWithout Principal Component Analysis (3D)')
print(f'True origins: {test_origins}')
print(f'Origins predicted with Decision Trees: {predicted_origins}')
print(f'Accuracy (mean accuracy): {accuracy:.2f}')
