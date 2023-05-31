# greencoffee project documentation

## adjust_data.py

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

### adjust_data

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

### drop_duplicates

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

### drop_missing

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

## price_score.py

    Script for investigating the dependence of green coffee price on its score

    Creates a Linear Regression model for the data, visualizes the data and the results

    Requires installation of 'pandas', 'matplotlib', 'scikit-learn'

## origins.py

    Module for simplifying possible coffee origins

    Requires installation of 'pandas'

    Functions
    ---------
    simplify_origins
        Takes a Series/Data Frame (selected features) and a Series (coffee names) as input, returns a Series/Data Frame
        and a Series
        Simplifies coffee names to countries, then to continents, then to continent ids

    Notes
    -----
    For reference, print printable_origins_map (continent: continent_id dictionary)

### simplify_origins

    Simplifies possible origins - takes into account continents rather than countries, changes alphabetical data into
    numerical data

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

## origin_weight_price.py

    Script for classifying green coffee origin based on its price and weight

    Transforms the data using 'origins.py', visualizes the data, creates a k-Nearest Neighbors Classifier and fits it on the
    training set, tests the model on the testing set

    Requires installation of 'pandas', 'matplotlib', 'scikit-learn'

## origin_profile.py

    Script for classifying green coffee origin based on its sensory profile

    Transforms the data using 'origins.py', vectorizes the sensory profiles using Count Vectorizer with Nltk library
    (tokenization and stemming), creates Multinomial and Complement Naive Bayes Classifiers and fits them on the training
    set, tests the models on the testing set

    Requires installation of 'pandas', 'nltk', 'scikit-learn'

### check

    Checks for stop words and punctuation

    Parameters
    ----------
    token: str
        Token to be checked

    Returns
    -------
    bool
        Whether the token is a stop word or a punctuation character

### tokenize

    Tokenizes the text, stems the words, removes stop words

    Parameters
    ----------
    text: str
        Text to be tokenized

    Returns
    -------
    list[str]
        Stemmed words of the input text

## main.py

    Main script of the project

    Transforms 'data.csv' and 'test_data.csv' using 'adjust_data.py'
    Runs 'price_score.py', 'origin_weight_price.py', 'origin_profile.py'

    Requires installation of 'pandas', 'matplotlib', 'scikit-learn', 'nltk'
