"""
Module for adjusting and acquiring data for further analysis

Requires installation of 'pandas'

Functions
---------
adjust_data
    Simplifies column labels, fixes data formatting and data types, then creates an adjusted csv file
drop_duplicates
    Drops data present in the training set from the testing set, then overwrites the latter file
drop_missing
    Removes rows with NaNs
get_data
    Gets relevant data from the adjusted data file(s)
"""

import pandas


def adjust_data(file_name: str) -> None:
    """
    Simplifies column labels, fixes data formatting and data types, then creates an adjusted csv file

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
    data = data.rename(columns={'SCA score': 'Score', 'Sensory profile': 'Profile', 'Approx. no of bags SPOT': 'Bags',
                                'Cena PLN/kg netto': 'Price'})
    # drop irrelevant columns
    data = data.drop(columns=['Price USD/kg', 'Price EUR/kg'])

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
    Drops data present in the training set from the testing set, then overwrites the latter file

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


def drop_missing(data_to_check: pandas.Series | pandas.DataFrame,
                 related_data: pandas.Series | pandas.DataFrame) \
        -> tuple[pandas.Series | pandas.DataFrame, pandas.Series | pandas.DataFrame]:
    """
    Removes rows with NaNs

    Adjusts Data Frames and Series already in use

    Parameters
    ----------
    data_to_check: pandas.Series | pandas.DataFrame
        Data with NaNs possibly present
    related_data: pandas.Series | pandas.DataFrame
        Data related to data_to_check

    Returns
    -------
    tuple[pandas.Series | pandas.DataFrame, pandas.Series | pandas.DataFrame]
        Input data without NaN-including rows
    """
    
    # remove rows with NaNs from the data given
    if isinstance(data_to_check, pandas.Series):
        for index, missing in data_to_check.isna().items():
            if missing:
                data_to_check = data_to_check.drop(index)
                related_data = related_data.drop(index)
    elif isinstance(data_to_check, pandas.DataFrame):
        for index, series in data_to_check.iterrows():
            if series.hasnans:
                data_to_check = data_to_check.drop(index)
                related_data = related_data.drop(index)
    
    # reset indices
    data_to_check = data_to_check.reset_index(drop=True)
    related_data = related_data.reset_index(drop=True)

    return data_to_check, related_data


def get_data(features_names: str | list[str], target_name: str, test: bool = True) \
        -> tuple[pandas.Series | pandas.DataFrame, pandas.Series] \
        | tuple[pandas.Series | pandas.DataFrame, pandas.Series, pandas.Series | pandas.DataFrame, pandas.Series]:
    """
    Gets relevant data from the adjusted data file(s)

    Parameters
    ----------
    features_names: str | list[str]
        Name(s) of column(s) containing the feature(s) of interest
    target_name: str
        Name of a column containing the target of interest
    test: bool, default True
        Whether to get testing data

    Returns
    -------
    tuple[pandas.Series | pandas.DataFrame, pandas.Series] \
    | tuple[pandas.Series | pandas.DataFrame, pandas.Series, pandas.Series | pandas.DataFrame, pandas.Series]
        Training features and target, optionally testing features and target
    """

    # read relevant training data from the adjusted file
    data = pandas.read_csv('adjusted_data.csv')
    train_features = data[features_names]
    train_target = data[target_name]
    # drop rows with missing values using drop_missing
    train_features, train_target = drop_missing(train_features, train_target)

    if test:

        # read relevant testing data from the adjusted file
        test_data = pandas.read_csv('adjusted_test_data.csv')
        test_features = test_data[features_names]
        test_target = test_data[target_name]
        # drop rows with missing values using drop_missing
        test_features, test_target = drop_missing(test_features, test_target)

        return train_features, train_target, test_features, test_target

    else:

        return train_features, train_target
