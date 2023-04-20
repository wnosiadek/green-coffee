# read csv data
# adjust the data where necessary
# write the data to a new csv file

import pandas

# read data from the provided file
data = pandas.read_csv('data.csv')

# column labels
print(data.columns)
# ['Weight', 'Coffee', 'Process', 'SCA score', 'Sensory profile',
#        'Approx. no of bags SPOT', 'Cena PLN/kg netto', 'Price USD/kg',
#        'Price EUR/kg']

# rename column labels
data.rename(columns={'SCA score': 'Score', 'Sensory profile': 'Profile',
                     'Approx. no of bags SPOT': 'Bags',
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
data.to_csv('adjusted_data.csv')
