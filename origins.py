"""
Module for simplifying possible coffee origins

Requires installation of 'pandas'

Functions
---------
simplify_origins
    Simplifies coffee names to countries, then to continents, then to continent ids

Notes
-----
For reference, print printable_origins_map (continent: continent_id dictionary)
"""

import pandas
import adjust_data

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
# create a printable version of the mapping for later reference
printable_origins_map = {continent: continent_id
                         for continent_id, continent
                         in enumerate(['Africa', 'Asia & Oceania', 'Mexico & Central America', 'South America'])}
# {'Africa': 0, 'Asia & Oceania': 1, 'Mexico & Central America': 2, 'South America': 3}


# use the above to transform given data
def simplify_origins(features: pandas.Series | pandas.DataFrame,
                     origins: pandas.Series) \
        -> tuple[pandas.Series | pandas.DataFrame, pandas.Series]:
    """
    Simplifies coffee names to countries, then to continents, then to continent ids

    Parameters
    ----------
    features: pandas.Series | pandas.DataFrame
        Chosen coffee characteristics
    origins: pandas.Series
        Coffee names, including the countries of origin

    Returns
    -------
    tuple[pandas.Series | pandas.DataFrame, pandas.Series]
        Input characteristics and origins simplified with origins_map
    """

    # extract countries from coffee names
    origins = origins.str.partition()[0]
    # ['Brazil' 'Colombia' 'Costa' 'Dom.' 'Ethiopia' 'Honduras' 'Indonesia' 'Kenya' 'Malawi' 'Mexico' 'Nicaragua'
    #  'NicaraguaSHB' 'Panama' 'Peru' 'PNG' 'Rwanda' 'Tanzania' 'Uganda' 'Yemen' 'Vietnam']
    # change Costa to Costa Rica, Dom. to Dominican Republic, PNG to Papua New Guinea
    origins = origins.replace({'Costa': 'Costa Rica', 'Dom.': 'Dominican Republic', 'PNG': 'Papua New Guinea'})
    # there are some mistakes (e.g. NicaraguaSHB) but they are inevitable

    # map countries to appropriate ids
    origins = origins.map(origins_map)
    # drop any mistakes using 'adjust_data.py'
    origins, features = adjust_data.drop_missing(origins, features)

    return features, origins
