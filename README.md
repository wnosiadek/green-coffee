# greencoffee

data analysis / machine learning basics

repository created for practicing data analysis and machine learning methods using an example of personal interest - green coffee

supervised learning, classification and regression, natural language processing, dimensionality reduction

language: Python, libraries: Pandas, Matplotlib, Scikit-learn, Nltk, Scipy, Numpy

## branches

### main

basic version

### main_pca

\+ dimensionality reduction

## documentation

#### DOCS.md

## data

#### data.csv
#### test_data.csv
#### adjusted_data.csv
#### adjusted_test_data.csv

## scripts and modules

### adjust_data.py

adjusting data before or during analysis

### price_score.py

investigating the dependence of green coffee price on its score

algorithms: Linear Regression

### origins.py

simplifying possible coffee origins

### processes.py

simplifying possible processing methods

### profiles.py

simplifying possible sensory profiles

algorithms: Snowball Stemmer, Count Vectorizer

### origin_weight_score_price.py

classifying green coffee origin based on its price, score and weight

algorithms: Min-Max Scaler, Principal Component Analysis, Decision Tree Classifier

### origin_profile.py

classifying green coffee origin based on its sensory profile

algorithms: Multinomial and Complement Naive Bayes Classifiers

### process_weight_score_price.py

classifying green coffee processing method based on its price, score and weight

algorithms: Min-Max Scaler, Principal Component Analysis, k-Nearest Neighbors and Fixed-Radius Near Neighbors Classifiers

### process_profile.py

classifying green coffee processing method based on its sensory profile

algorithms: Support Vector Classifier

### main_pca.py

main script of the project

## plots

#### price_score.png
#### origin_weight_score_price.png
#### origin_weight_score_price_tree.png
#### process_weight_score_price.png
