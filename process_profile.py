"""
Script for classifying green coffee processing method based on its sensory profile

Transforms the data using 'processes.py', vectorizes the sensory profiles using 'profiles.py', creates a Support Vector
Classifier and fits it on the training set, tests the model on the testing set

Requires installation of 'pandas', 'nltk', 'scipy', 'scikit-learn'
"""

import data
import processes
import profiles
from sklearn.svm import SVC

print('\nClassifying green coffee processing method based on its sensory profile...')

# get relevant training and testing data using 'data.py'
train_profiles, train_processes, test_profiles, test_processes = data.get_data('Profile', 'Process')

# transform training and testing data using 'processes.py'
# (simplify processing method names to integer ids, print the mapping for reference)
print()
train_profiles, train_processes, test_profiles, test_processes \
    = processes.simplify_processes(train_profiles, train_processes, test_profiles, test_processes, print_map=True)

# vectorize the profiles using 'profiles.py'
train_profiles_vector, test_profiles_vector = profiles.vectorize(train_profiles, test_profiles)

# create a Support Vector Classifier for the training data
classifier = SVC()
classifier.fit(train_profiles_vector, train_processes.to_numpy())

# make predictions on the testing data
predicted_processes = classifier.predict(test_profiles_vector)
accuracy = classifier.score(test_profiles_vector, test_processes.to_numpy())

# the results
print(f'\nTrue processing methods: {test_processes.to_numpy()}')
print(f'Processing methods predicted with Support Vector Machines: {predicted_processes}')
print(f'Accuracy (mean accuracy): {accuracy:.2f}')
