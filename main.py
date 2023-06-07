"""
Main script of the project

Transforms 'data.csv' and 'test_data.csv' using 'adjust_data.py'
Runs 'price_score.py', 'origin_weight_price.py', 'origin_profile.py', 'process_profile.py', 'process_score_price.py'

Requires installation of 'pandas', 'matplotlib', 'scikit-learn', 'nltk', 'scipy'
"""

import adjust_data

# transform data files using 'adjust_data.py'
adjust_data.adjust_data('data.csv')
adjust_data.adjust_data('test_data.csv')
adjust_data.drop_duplicates('adjusted_data.csv', 'adjusted_test_data.csv')

import price_score

# output of 'price_score.py'
'''
Investigating the dependence of green coffee price on its score...

Model obtained with Linear Regression: y = 6x + -459
Accuracy (coefficient of determination): R2 = 0.48
'''
# pop-up plot: 'price_score.png'

import origin_weight_price

# output of 'origin_weight_price.py'
'''
Classifying green coffee origin based on its price and weight...

{'Africa': 0, 'Asia & Oceania': 1, 'Mexico & Central America': 2, 'South America': 3}

True origins: [3 3 3 3 3 2 2 2 3 0 0]
Origins predicted with Decision Trees: [3. 3. 3. 3. 3. 3. 2. 2. 3. 0. 0.]
Accuracy (mean accuracy): 0.91
'''
# pop-up plots: 'origin_weight_price.png', 'origin_weight_price_tree.png'

import origin_profile

# output of 'origin_profile.py'
'''
Classifying green coffee origin based on its sensory profile...

{'Africa': 0, 'Asia & Oceania': 1, 'Mexico & Central America': 2, 'South America': 3}

True origins: [3 3 3 3 3 2 3 0 0]
Origins predicted with Multinomial Naive Bayes: [3. 3. 2. 3. 3. 3. 3. 0. 0.]
Accuracy (mean accuracy): 0.78

True origins: [3 3 3 3 3 2 3 0 0]
Origins predicted with Complement Naive Bayes: [3. 3. 2. 2. 2. 3. 3. 0. 0.]
Accuracy (mean accuracy): 0.56
'''

import process_profile

# output of 'process_profile.py'
'''
Classifying green coffee processing method based on its sensory profile...

{'washed': 0, 'natural': 1, 'pulped natural': 2, 'wet hulled': 3, 'anaerobic': 4}

True processing methods: [2 1 1 0 0 0 0 0 1]
Processing methods predicted with Support Vector Machines: [0. 0. 1. 0. 0. 0. 0. 1. 0.]
Accuracy (mean accuracy): 0.56
'''

import process_score_price

# output of 'process_score_price.py'
'''
Classifying green coffee processing method based on its price and score...

{'washed': 0, 'natural': 1, 'pulped natural': 2, 'anaerobic': 3}

True processing methods: [2 0 1]
Processing methods predicted with k-Nearest Neighbors: [1. 1. 0.]
Accuracy (mean accuracy): 0.00

True processing methods: [2 0 1]
Processing methods predicted with Fixed-Radius Near Neighbors: [0. 1. 0.]
Accuracy (mean accuracy): 0.00
'''
# pop-up plot: 'process_score_price.png'
