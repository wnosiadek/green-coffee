# functions adjusting data for analysis

import pandas


def adjust_data(file_name):

  # read csv data
  # adjust the data where necessary
  # write the data to a new csv file

  # read data from the provided file
  data = pandas.read_csv(file_name)

  # column labels
  print(data.columns)
  # ['Weight', 'Coffee', 'Process', 'SCA score', 'Sensory profile', 'Approx. no of bags SPOT',
  #  'Cena PLN/kg netto', 'Price USD/kg', 'Price EUR/kg']

  # rename column labels
  data.rename(columns={'SCA score': 'Score', 'Sensory profile': 'Profile', 'Approx. no of bags SPOT': 'Bags',
                       'Cena PLN/kg netto': 'Price'}, inplace=True)
  # drop irrelevant columns
  data.drop(columns=['Price USD/kg', 'Price EUR/kg'], inplace=True)

  # new column labels
  print(data.columns)
  # ['Weight', 'Coffee', 'Process', 'Score', 'Profile', 'Bags', 'Price']

  # adjust the data

  # remove units from weights
  # convert numeric strings to integers
  data['Weight'] = data['Weight'].str.removesuffix(' kg')
  data['Weight'] = pandas.to_numeric(data['Weight'])
  # replace commas with periods in scores
  # convert numeric strings to floats
  # replace alphabetic strings and dashes with nans
  data['Score'] = data['Score'].astype(str)
  data['Score'] = data['Score'].str.replace(',', '.')
  data['Score'] = pandas.to_numeric(data['Score'], errors='coerce')
  # replace dashes with nans
  data['Profile'] = data['Profile'].mask(data['Profile'] == '-')
  # ensure that bags values are integers
  data['Bags'] = pandas.to_numeric(data['Bags'])
  # replace commas with periods in prices
  # convert numeric strings to floats
  data['Price'] = data['Price'].str.replace(',', '.')
  data['Price'] = pandas.to_numeric(data['Price'])

  # write the data to a new file
  data.to_csv('adjusted_' + file_name, index=False)

  
def drop_duplicates(train_file_name, test_file_name):
  
  # read csv data for training and testing
  # drop samples present in both sets from the testing set
  # overwrite the testing csv data file

  # read data from the provided files
  train_data = pandas.read_csv(train_file_name)
  test_data = pandas.read_csv(test_file_name)

  # drop duplicated samples from the testing data
  for index, sample in test_data['Coffee'].items():
      if sample in train_data['Coffee'].values:
          test_data = test_data.drop(index)
  test_data = test_data.reset_index(drop=True)

  # overwrite the testing data
  test_data.to_csv(test_file_name, index=False)
