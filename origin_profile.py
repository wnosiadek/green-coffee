"""
Script for classifying green coffee origin based on its sensory profile

Transforms the data using 'origins.py', vectorizes the sensory profiles using 'profiles.py', creates Multinomial and 
Complement Naive Bayes Classifiers and fits them on the training set, tests the models on the testing set

Requires installation of 'pandas', 'nltk', 'scipy', 'scikit-learn'
"""

import pandas
import adjust_data
import origins
import profiles
from sklearn.naive_bayes import MultinomialNB, ComplementNB

print('\nClassifying green coffee origin based on its sensory profile...')

# read relevant training data from the adjusted file
data = pandas.read_csv('adjusted_data.csv')
train_profiles = data['Profile']
train_origins = data['Coffee']
# drop rows with missing values using 'adjust_data.py'
train_profiles, train_origins = adjust_data.drop_missing(train_profiles, train_origins)

# read relevant testing data from the adjusted file
test_data = pandas.read_csv('adjusted_test_data.csv')
test_profiles = test_data['Profile']
test_origins = test_data['Coffee']
# drop rows with missing values using 'adjust_data.py'
test_profiles, test_origins = adjust_data.drop_missing(test_profiles, test_origins)

# transform training and testing data using 'origins.py'
# (simplify possible origins, take into account continents rather than countries, change alphabetical data into
#  numerical data)
train_profiles, train_origins = origins.simplify_origins(train_profiles, train_origins)
test_profiles, test_origins = origins.simplify_origins(test_profiles, test_origins)
# print the mapping for reference
print(f'\n{origins.printable_origins_map}')

# vectorize the profiles using 'profiles.py'
train_profiles_vector, test_profiles_vector = profiles.vectorize(train_profiles, test_profiles)

# create Multinomial and Complement Naive Bayes Classifiers for the training data
multinomial_classifier = MultinomialNB()
multinomial_classifier.fit(train_profiles_vector, train_origins.to_numpy())
complement_classifier = ComplementNB()
complement_classifier.fit(train_profiles_vector, train_origins.to_numpy())

# make predictions on the testing data
multinomial_predicted_origins = multinomial_classifier.predict(test_profiles_vector)
multinomial_accuracy = multinomial_classifier.score(test_profiles_vector, test_origins.to_numpy())
complement_predicted_origins = complement_classifier.predict(test_profiles_vector)
complement_accuracy = complement_classifier.score(test_profiles_vector, test_origins.to_numpy())

# the results
print(f'\nTrue origins: {test_origins.to_numpy()}')
print(f'Origins predicted with Multinomial Naive Bayes: {multinomial_predicted_origins}')
print(f'Accuracy (mean accuracy): {multinomial_accuracy:.2f}')
print(f'\nTrue origins: {test_origins.to_numpy()}')
print(f'Origins predicted with Complement Naive Bayes: {complement_predicted_origins}')
print(f'Accuracy (mean accuracy): {complement_accuracy:.2f}')
