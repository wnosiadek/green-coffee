"""
Main script of the project

Transforms 'data.csv' and 'test_data.csv' using 'adjust_data.py'
Runs 'price_score.py', 'origin_weight_price.py', 'origin_profile.py'

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
Origins predicted with k-Nearest Neighbors: [3. 3. 3. 3. 3. 2. 2. 2. 2. 0. 0.]
Accuracy (mean accuracy): 0.91
'''
# pop-up plot: 'origin_weight_price.png'

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
