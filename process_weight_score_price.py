"""
Script for classifying green coffee processing method based on its price, score and weight

Transforms the data using 'processes.py', scales the data with Min-Max Scaler and reduces its dimensionality with
Principal Component Analysis, visualizes the data, creates k-Nearest Neighbors and Fixed-Radius Near Neighbors
Classifiers and fits them on the training set, tests the models on the testing set

Requires installation of 'pandas', 'scikit-learn', 'numpy', 'matplotlib'
"""

import data
import processes
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

print('\nClassifying green coffee processing method based on its price, score and weight...')

# get relevant training and testing data using 'data.py'
train_features, train_processes, test_features, test_processes \
    = data.get_data(['Weight', 'Score', 'Price'], 'Process')

# transform training and testing data using 'processes.py'
# (simplify processing method names to integer ids, print the mapping for reference)
train_features, train_processes, test_features, test_processes \
    = processes.simplify_processes(train_features, train_processes, test_features, test_processes, print_map=True)

# convert the data to numpy arrays
train_features = train_features.to_numpy()
train_processes = train_processes.to_numpy()
test_features = test_features.to_numpy()
test_processes = test_processes.to_numpy()

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
pyplot.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=train_processes, cmap='gist_rainbow')
pyplot.colorbar(ticks=train_processes)
pyplot.xlabel(f'I PC: [Weight, Score, Price] = {pca_components[0]}')
pyplot.ylabel(f'II PC: [Weight, Score, Price] = {pca_components[1]}')
pyplot.title('Green coffee processing method classification')
pyplot.scatter(test_features_pca[:, 0], test_features_pca[:, 1], c='black')
pyplot.show()

# create k-Nearest Neighbors and Fixed-Radius Near Neighbors Classifiers for the training data
k_classifier_pca = KNeighborsClassifier()
k_classifier_pca.fit(train_features_pca, train_processes)
radius_classifier_pca = RadiusNeighborsClassifier()
radius_classifier_pca.fit(train_features_pca, train_processes)

# make predictions on the testing data
k_predicted_processes_pca = k_classifier_pca.predict(test_features_pca)
k_accuracy_pca = k_classifier_pca.score(test_features_pca, test_processes)
radius_predicted_processes_pca = radius_classifier_pca.predict(test_features_pca)
radius_accuracy_pca = radius_classifier_pca.score(test_features_pca, test_processes)

# the results
print('\nWith Principal Component Analysis (2D)')
print(f'\nTrue processing methods: {test_processes}')
print(f'Processing methods predicted with k-Nearest Neighbors: {k_predicted_processes_pca}')
print(f'Accuracy (mean accuracy): {k_accuracy_pca:.2f}')
print(f'\nTrue processing methods: {test_processes}')
print(f'Processing methods predicted with Fixed-Radius Near Neighbors: {radius_predicted_processes_pca}')
print(f'Accuracy (mean accuracy): {radius_accuracy_pca:.2f}')

# the same but without PCA

# create k-Nearest Neighbors and Fixed-Radius Near Neighbors Classifiers for the training data
k_classifier = KNeighborsClassifier()
k_classifier.fit(train_features, train_processes)
radius_classifier = RadiusNeighborsClassifier()
radius_classifier.fit(train_features, train_processes)

# make predictions on the testing data
k_predicted_processes = k_classifier.predict(test_features)
k_accuracy = k_classifier.score(test_features, test_processes)
radius_predicted_processes = radius_classifier.predict(test_features)
radius_accuracy = radius_classifier.score(test_features, test_processes)

# the results
print('\nWithout Principal Component Analysis (3D)')
print(f'\nTrue processing methods: {test_processes}')
print(f'Processing methods predicted with k-Nearest Neighbors: {k_predicted_processes}')
print(f'Accuracy (mean accuracy): {k_accuracy:.2f}')
print(f'\nTrue processing methods: {test_processes}')
print(f'Processing methods predicted with Fixed-Radius Near Neighbors: {radius_predicted_processes}')
print(f'Accuracy (mean accuracy): {radius_accuracy:.2f}')
