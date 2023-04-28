# classify origins based on price and weight
# use k-nearest neighbors classifier
from typing import Any

import pandas
from matplotlib import pyplot
from pandas import Series, DataFrame
from sklearn.neighbors import KNeighborsClassifier

# read relevant training data from the adjusted file
data = pandas.read_csv('adjusted_data.csv')
train_features = data[['Weight', 'Price']]
train_origins = data['Coffee']

# read relevant testing data from the adjusted file
test_data = pandas.read_csv('adjusted_test_data.csv')
test_features = test_data[['Weight', 'Price']]
test_origins = test_data['Coffee']
# drop samples present also in the training data
for index, sample in test_origins.items():
    if sample in train_origins.values:
        test_origins = test_origins.drop(index)
        test_features = test_features.drop(index)
test_origins = test_origins.reset_index(drop=True)
test_features = test_features.reset_index(drop=True)

# simplify origins
# take into account continents rather than countries
# change alphabetical data into numerical data

# possible continents
africa = ['Burundi', 'Cameroon', "Côte d'Ivoire", 'Democratic Republic of Congo', 'Ethiopia', 'Guinea', 'Kenya',
          'Madagascar', 'Rwanda', 'Tanzania', 'Togo', 'Uganda']
asia = ['India', 'Indonesia', "Lao People's Democratic Republic", 'Papua New Guinea', 'Philippines', 'Thailand',
        'Vietnam', 'Yemen']
c_america = ['Costa Rica', 'Cuba', 'Dominican Republic', 'El Salvador', 'Guatemala', 'Haiti', 'Honduras', 'Mexico',
             'Nicaragua', 'Panama']
s_america = ['Bolivia', 'Brazil', 'Colombia', 'Ecuador', 'Peru', 'Venezuela']

# assign an integer id to each continent, and then to each country on that continent
origins_map = {country: continent_id
               for continent_id, continent in enumerate([africa, asia, c_america, s_america])
               for country in continent}
print({continent: continent_id for continent_id, continent
       in enumerate(['Africa', 'Asia & Oceania', 'Mexico & Central America', 'South America'])})
# {'Africa': 0, 'Asia & Oceania': 1, 'Mexico & Central America': 2, 'South America': 3}

# use the above to transform the data
def simplify_origins(features, origins):

    # extract countries from coffee names
    origins = origins.str.partition()[0]
    print(origins.unique())
    # ['Brazil' 'Colombia' 'Costa' 'Dom.' 'Ethiopia' 'Honduras' 'Indonesia'
    #  'Kenya' 'Malawi' 'Mexico' 'Nicaragua' 'NicaraguaSHB' 'Panama' 'Peru'
    #  'PNG' 'Rwanda' 'Tanzania' 'Uganda' 'Yemen' 'Vietnam']
    # change Costa to Costa Rica, Dom. to Dominican Republic, PNG to Papua New Guinea
    origins = origins.replace({'Costa': 'Costa Rica', 'Dom.': 'Dominican Republic', 'PNG': 'Papua New Guinea'})
    # there are some mistakes (eg. NicaraguaSHB) but they are inevitable

    # map countries to appropriate ids
    origins = origins.map(origins_map)
    # drop any mistakes
    for index, mistake in origins.isna().items():
        if mistake:
            origins = origins.drop(index)
            features = features.drop(index)
    origins = origins.reset_index(drop=True)
    features = features.reset_index(drop=True)

    return features, origins


# transform training and testing data
train_features, train_origins = simplify_origins(train_features, train_origins)
test_features, test_origins = simplify_origins(test_features, test_origins)

# plot the data
pyplot.scatter('Weight', 'Price', data=train_features, c=train_origins, cmap='gist_rainbow')
pyplot.colorbar(ticks=train_origins)
pyplot.xlabel('Weight kg')
pyplot.ylabel('Price PLN/kg')
pyplot.title('Green coffee origin classification')
pyplot.scatter('Weight', 'Price', data=test_features, c='black')
pyplot.show()

# convert the data to numpy arrays
train_features = train_features.to_numpy()
train_origins = train_origins.to_numpy()
test_features = test_features.to_numpy()
test_origins = test_origins.to_numpy()

# create a k-nearest neighbors vote for the training data
vote = KNeighborsClassifier()
vote.fit(train_features, train_origins)

# make predictions on the testing data
predicted_origins = vote.predict(test_features)
accuracy = vote.score(test_features, test_origins)

# the results
print(f'True origins: {test_origins}')
print(f'Predicted origins: {predicted_origins}')
print(f'Accuracy: {accuracy:.2f}')
