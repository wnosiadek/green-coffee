# read the xlsx data
# adjust the data where necessary
# write the data to a new xlsx

import pandas

# read data from the provided file
data = pandas.read_excel('data.xlsx')

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

# convert strings to ints, floats and NaNs where necessary

data['Weight'] = data['Weight'].str.removesuffix(' kg')
data['Weight'] = pandas.to_numeric(data['Weight'])

data['Score'] = data['Score'].astype(str)
data['Score'] = data['Score'].str.replace(',', '.')
data['Score'] = pandas.to_numeric(data['Score'], errors='coerce')

data['Profile'] = data['Profile'].mask(data['Profile'] == '-')

data['Bags'] = pandas.to_numeric(data['Bags'])

data['Price'] = data['Price'].str.replace(',', '.')
data['Price'] = pandas.to_numeric(data['Price'])

# write the data to a new file
data.to_excel('adjusted_data.xlsx')
