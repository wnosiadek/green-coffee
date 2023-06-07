# greencoffee project documentation

## adjust_data.py

    Module for adjusting data before or during analysis

    Requires installation of 'pandas'

    Functions
    ---------
    adjust_data
        Simplifies column labels, fixes data formatting and data types, then creates an adjusted csv file
    drop_duplicates
        Drops data present in the training set from the testing set, then overwrites the latter file
    drop_missing
        Removes rows with NaNs

### adjust_data

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

### drop_duplicates

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

### drop_missing

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
        Simplifies coffee names to countries, then to continents, then to continent ids

    Notes
    -----
    For reference, print printable_origins_map (continent: continent_id dictionary)

### simplify_origins

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

## processes.py

    Module for simplifying possible processing methods

    Requires installation of 'pandas'

    Functions
    ---------
    simplify_processes
        Simplifies processing method names to integer ids

### simplify_processes

    Simplifies processing method names to integer ids

    Parameters
    ----------
    train_features: pandas.Series | pandas.DataFrame
        Chosen coffee characteristics for training
    train_processes: pandas.Series
        Processing methods for training
    test_features: pandas.Series | pandas.DataFrame
        Chosen coffee characteristics for testing
    test_processes: pandas.Series
        Processing methods for testing
    print_map: bool, default True
        Whether to print the {process: process_id} map

    Returns
    -------
    tuple[pandas.Series | pandas.DataFrame, pandas.Series, pandas.Series | pandas.DataFrame, pandas.Series]
        Input characteristics and processing methods simplified with the {process: process_id} map

## profiles.py

    Module for simplifying possible sensory profiles

    Uses Snowball Stemmer, Count Vectorizer

    Requires installation of 'nltk', 'pandas', 'scipy', 'scikit-learn'

    Functions
    ---------
    check
        Checks for stop words and punctuation
    tokenize
        Tokenizes the text, stems the words, removes stop words
    vectorize
        Vectorizes the training and testing profiles

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

### vectorize

    Vectorizes the training and testing profiles

    Parameters
    ----------
    train_profiles: pandas.Series
        Sensory profiles for training
    test_profiles: pandas.Series
        Sensory profiles for testing

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]
        Profiles vectorized using Count Vectorizer with tokenize

## origin_weight_price.py

    Script for classifying green coffee origin based on its price and weight

    Transforms the data using 'origins.py', visualizes the data, creates a Decision Tree Classifier and fits it on the 
    training set, visualizes the tree, tests the model on the testing set

    Requires installation of 'pandas', 'matplotlib', 'scikit-learn'

## origin_profile.py

    Script for classifying green coffee origin based on its sensory profile

    Transforms the data using 'origins.py', vectorizes the sensory profiles using 'profiles.py', creates Multinomial and 
    Complement Naive Bayes Classifiers and fits them on the training set, tests the models on the testing set

    Requires installation of 'pandas', 'nltk', 'scipy', 'scikit-learn'

## process_score_price.py

    Script for classifying green coffee processing method based on its price and score

    Transforms the data using 'processes.py', visualizes the data, creates k-Nearest Neighbors and Fixed-Radius Near
    Neighbors Classifiers and fits them on the training set, tests the models on the testing set

    Requires installation of 'pandas', 'matplotlib', 'scikit-learn'

## process_profile.py

    Script for classifying green coffee processing method based on its sensory profile

    Transforms the data using 'processes.py', vectorizes the sensory profiles using 'profiles.py', creates a Support Vector
    Classifier and fits it on the training set, tests the model on the testing set

    Requires installation of 'pandas', 'nltk', 'scipy', 'scikit-learn'

## main.py

    Main script of the project

    Transforms 'data.csv' and 'test_data.csv' using 'adjust_data.py'
    Runs 'price_score.py', 'origin_weight_price.py', 'origin_profile.py', 'process_profile.py', 'process_score_price.py'

    Requires installation of 'pandas', 'matplotlib', 'scikit-learn', 'nltk', 'scipy'
