# green-coffee

data analysis / machine learning basics

repository created for practicing basic data analysis and machine learning methods using an example of personal interest: green coffee

programming language: **Python**, used libraries: **Pandas**, **Matplotlib**, **Scikit-learn**

## data files

- **data.csv** / **adjusted_data.csv**: data files for training purposes before / after the necessary adjustments
- **test_data.csv** / **adjusted_test_data.csv**: data files for testing purposes before / after the necessary adjustments

## scripts and modules

### adjust_data.py

module for adjusting data

contains a function **adjust_data**, taking a csv file name as input and returning None

**adjust_data** simplifies column labels, fixes data formatting and data types, then creates an adjusted csv file

### price_and_score.py

script for investingating the dependence of green coffee price on its score

creates a **Linear Regression** model for the data, visualizes the data and the results

### origin.py

script for classifying green coffee origins based on its price and weight

translates countries of origin to continents of origin, visualizes the data, creates and fits a **K Nearest Neighbors Classifier**, tests the model on new data
