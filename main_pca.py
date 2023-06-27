"""
Main script of the project

Transforms 'data.csv' and 'test_data.csv' using 'data.py'
Runs 'price_score.py', 'origin_weight_score_price.py', 'origin_profile.py', 'process_weight_score_price.py', 
'process_profile.py'

Requires installation of 'pandas', 'matplotlib', 'scikit-learn', 'nltk', 'scipy', 'numpy'
"""

import data

# transform data files using 'data.py'
data.adjust_data('data.csv')
data.adjust_data('test_data.csv')
data.drop_duplicates('adjusted_data.csv', 'adjusted_test_data.csv')

import price_score

# output of 'price_score.py'
'''
Investigating the dependence of green coffee price on its score...

Model obtained with Linear Regression: y = 6x + -459
Accuracy (coefficient of determination): R2 = 0.48
'''
# pop-up plot: 'price_score.png'

import origin_weight_score_price

# output of 'origin_weight_score_price.py'
'''
Classifying green coffee origin based on its price, score and weight...

{'Africa': 0, 'Asia & Oceania': 1, 'Mexico & Central America': 2, 'South America': 3}

Reducing the dimensionality...

Principal components in [Weight, Score, Price] space:
[[-0.55  0.70  0.45]
 [ 0.83  0.53  0.19]]
Variance explained by each component (0.92 in total):
[0.62 0.30]

With Principal Component Analysis (2D)
True origins: [2 2 2 2 0 0 0 0 0]
Origins predicted with Decision Trees: [2 3 3 3 0 0 3 3 0]
Accuracy (mean accuracy): 0.44

Without Principal Component Analysis (3D)
True origins: [2 2 2 2 0 0 0 0 0]
Origins predicted with Decision Trees: [2 2 2 2 0 0 0 1 0]
Accuracy (mean accuracy): 0.89
'''
# pop-up plots: 'origin_weight_score_price.png', 'origin_weight_score_price_tree.png'

import origin_profile

# output of 'origin_profile.py'
'''
Classifying green coffee origin based on its sensory profile...

{'Africa': 0, 'Asia & Oceania': 1, 'Mexico & Central America': 2, 'South America': 3}

True origins: [3 3 3 2 2 2 2 1 1 2 2 0 0 0 0 0]
Origins predicted with Multinomial Naive Bayes: [3. 3. 3. 0. 2. 3. 3. 2. 3. 2. 3. 0. 0. 0. 0. 3.]
Accuracy (mean accuracy): 0.56

True origins: [3 3 3 2 2 2 2 1 1 2 2 0 0 0 0 0]
Origins predicted with Complement Naive Bayes: [3. 2. 2. 2. 2. 2. 3. 2. 1. 2. 3. 0. 0. 0. 0. 1.]
Accuracy (mean accuracy): 0.62
'''

import process_weight_score_price

# output of 'process_weight_score_price.py'
'''
Classifying green coffee processing method based on its price, score and weight...

{'washed': 0, 'natural': 1, 'pulped natural': 2, 'anaerobic': 3}

Reducing the dimensionality...

Principal components in [Weight, Score, Price] space:
[[-0.50  0.71  0.49]
 [ 0.86  0.49  0.16]]
Variance explained by each component (0.92 in total):
[0.62 0.30]

With Principal Component Analysis (2D)

True processing methods: [1 0 0 0 0 1 0 0 0]
Processing methods predicted with k-Nearest Neighbors: [1. 0. 0. 0. 1. 0. 0. 0. 0.]
Accuracy (mean accuracy): 0.78

True processing methods: [1 0 0 0 0 1 0 0 0]
Processing methods predicted with Fixed-Radius Near Neighbors: [0. 0. 0. 0. 0. 0. 0. 0. 0.]
Accuracy (mean accuracy): 0.78

Without Principal Component Analysis (3D)

True processing methods: [1 0 0 0 0 1 0 0 0]
Processing methods predicted with k-Nearest Neighbors: [1. 0. 0. 0. 0. 0. 0. 0. 0.]
Accuracy (mean accuracy): 0.89

True processing methods: [1 0 0 0 0 1 0 0 0]
Processing methods predicted with Fixed-Radius Near Neighbors: [0. 0. 0. 0. 0. 0. 0. 0. 0.]
Accuracy (mean accuracy): 0.78
'''
# pop-up plot: 'process_weight_score_price.png'

import process_profile

# output of 'process_profile.py'
'''
Classifying green coffee processing method based on its sensory profile...

{'washed': 0, 'natural': 1, 'pulped natural': 2, 'wet hulled': 3, 'anaerobic': 4}

True processing methods: [1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
Processing methods predicted with Support Vector Machines: [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0.]
Accuracy (mean accuracy): 0.67
'''
