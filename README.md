# green-coffee

data analysis / machine learning basics

repository created for practicing basic data analysis and machine learning methods using an example of personal interest: green coffee

programming language: **Python**, used libraries: **Pandas**, **Matplotlib**, **Scikit-learn**, **Nltk**

## data files

- **data.csv** / **adjusted_data.csv**: data files for training purposes before / after the necessary adjustments using adjust_data.py
- **test_data.csv** / **adjusted_test_data.csv**: data files for testing purposes before / after the necessary adjustments using adjust_data.py

## scripts and modules

### adjust_data.py

module for adjusting data before or during analysis

functions:
- **adjust_data**
  - takes a csv file name (data) as input, returns None
  - simplifies column labels, fixes data formatting and data types, then creates an adjusted csv file
- **drop_duplicates**
  - takes two csv file names (training and testing data) as input, returns None
  - drops data present in the training set from the testing set, then overwrites the latter file
- **drop_missing**
  - takes two related data frames or series as input, returns the same data structures
  - removes rows with nans

### price_score.py

script for investingating the dependence of green coffee price on its score

creates a **Linear Regression** model for the data, visualizes the data and the results

### origins.py

module for simplifying possible coffee origins

functions:
- **simplify_origins**
  - takes a data frame (selected features) and a series (coffee names) as input, returns a data frame and a series
  - simplifies coffee names to countries, then to continents, then to continent ids

### origin_weight_price.py

script for classifying green coffee origins based on its price and weight

transforms the data using origins.py, visualizes the data, creates and fits a **K Nearest Neighbors Classifier** on the training set, tests the model on the testing set
