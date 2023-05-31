"""
Module for adjusting data before or during analysis

Requires installation of 'pandas'

Functions
---------
adjust_data
    Takes a csv file name (data) as input, returns None
    Simplifies column labels, fixes data formatting and data types, then creates an adjusted csv file
drop_duplicates
    Takes two csv file names (training and testing data) as input, returns None
    Drops data present in the training set from the testing set, then overwrites the latter file
drop_missing
    Takes two Series or a Series and a Data Frame as input, returns two Series or a Series and a Data Frame
    Removes rows with NaNs
"""

import pandas


def adjust_data(file_name: str) -> None:
    """
    Reads csv data, adjusts the data where necessary, writes the data to a new csv file

    Adjusts data files prior to use

    Parameters
    ----------
    file_name: str
        Name of the csv data file

    Returns
    -------
    None

    Notes
    -----
    Creates a new file named 'adjusted_' + file_name
    """

    # read data from the provided file
    data = pandas.read_csv(file_name)

    # column labels
    # ['Weight', 'Coffee', 'Process', 'SCA score', 'Sensory profile', 'Approx. no of bags SPOT',
    #  'Cena PLN/kg netto', 'Price USD/kg', 'Price EUR/kg']

    # rename column labels
    data.rename(columns={'SCA score': 'Score', 'Sensory profile': 'Profile', 'Approx. no of bags SPOT': 'Bags',
                         'Cena PLN/kg netto': 'Price'}, inplace=True)
    # drop irrelevant columns
    data.drop(columns=['Price USD/kg', 'Price EUR/kg'], inplace=True)

    # new column labels
    # ['Weight', 'Coffee', 'Process', 'Score', 'Profile', 'Bags', 'Price']

    # adjust the data

    # remove units from weights
    # convert numeric strings to integers
    data['Weight'] = data['Weight'].str.removesuffix(' kg')
    data['Weight'] = pandas.to_numeric(data['Weight'])
    
    # replace commas with periods in scores
    # convert numeric strings to floats
    # replace alphabetic strings and dashes with NaNs
    data['Score'] = data['Score'].astype(str)
    data['Score'] = data['Score'].str.replace(',', '.')
    data['Score'] = pandas.to_numeric(data['Score'], errors='coerce')
    
    # replace dashes with NaNs in profiles
    data['Profile'] = data['Profile'].mask(data['Profile'] == '-')
    
    # ensure that bags values are integers
    data['Bags'] = pandas.to_numeric(data['Bags'])
    
    # replace commas with periods in prices
    # convert numeric strings to floats
    data['Price'] = data['Price'].str.replace(',', '.')
    data['Price'] = pandas.to_numeric(data['Price'])

    # write the data to a new file
    data.to_csv('adjusted_' + file_name, index=False)

  
def drop_duplicates(train_file_name: str, test_file_name: str) -> None:
    """
    Reads csv data for training and testing, drops samples present in both sets from the testing set, overwrites the
    testing csv data file

    Adjusts data files prior to use
    To be used after adjust_data

    Parameters
    ----------
    train_file_name: str
        Name of the csv file with training data
    test_file_name: str
        Name of the csv file with testing data

    Returns
    -------
    None
    """

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


def drop_missing(data_to_check: pandas.Series,
                 related_data: pandas.Series | pandas.DataFrame)\
        -> tuple[pandas.Series, pandas.Series | pandas.DataFrame]:
    """
    Removes rows with NaNs

    Adjusts Data Frames and Series already in use

    Parameters
    ----------
    data_to_check: pandas.Series
        Data with NaNs possibly present
    related_data: pandas.Series | pandas.DataFrame
        Data related to data_to_check

    Returns
    -------
    tuple[pandas.Series, pandas.Series | pandas.DataFrame]
        Input data without NaN-including rows
    """
    
    # remove rows with NaNs from the data given
    for index, missing in data_to_check.isna().items():
        if missing:
            data_to_check = data_to_check.drop(index)
            related_data = related_data.drop(index)
    
    # reset indices
    data_to_check = data_to_check.reset_index(drop=True)
    related_data = related_data.reset_index(drop=True)

    return data_to_check, related_data
