"""
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
"""

from nltk.tokenize import wordpunct_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
import pandas
import scipy
from sklearn.feature_extraction.text import CountVectorizer


# define a stop words checker
def check(token: str) -> bool:
    """
    Checks for stop words and punctuation

    Parameters
    ----------
    token: str
        Token to be checked

    Returns
    -------
    bool
        Whether the token is a stop word or a punctuation character
    """

    return token not in stopwords.words('english') and token not in string.punctuation


# define a tokenizer
def tokenize(text: str) -> list[str]:
    """
    Tokenizes the text, stems the words, removes stop words

    Parameters
    ----------
    text: str
        Text to be tokenized

    Returns
    -------
    list[str]
        Stemmed words of the input text
    """

    # tokenize given text
    text_tokens = wordpunct_tokenize(text)

    # stem the words using Snowball Stemmer
    # remove stop words and punctuation using check
    stemmer = SnowballStemmer('english')
    text_tokens = [stemmer.stem(token) for token in text_tokens if check(token)]

    return text_tokens


# define a vectorizer
def vectorize(train_profiles: pandas.Series,
              test_profiles: pandas.Series)\
        -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
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
    """

    # create a Count Vectorizer for the training data
    # vectorize the training profiles
    # (tokenize the text using tokenize, count the occurrences for each coffee)
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
    train_profiles_vector = vectorizer.fit_transform(train_profiles)
    # vectorize the testing profiles
    test_profiles_vector = vectorizer.transform(test_profiles)

    # feature names corresponding to possible sensory notes
    # ['acid' 'almond' 'appl' 'apricot' 'bergamot' 'berri' 'biscuit' 'black' 'blackberri' 'blosom' 'blossom' 'blueberri'
    #  'bodi' 'brazil' 'brown' 'butter' 'camomill' 'caramel' 'cardamom' 'ceder' 'cherri' 'chewi' 'chocol' 'cinnamon'
    #  'citric' 'clove' 'cocoa' 'cranberri' 'creami' 'crisp' 'cup' 'currant' 'current' 'dark' 'date' 'dri' 'drop' 'earl'
    #  'eleg' 'fig' 'floral' 'flower' 'floweri' 'fruit' 'fudg' 'goosberri' 'gooseberri' 'grape' 'grapefruit' 'grass'
    #  'grey' 'grill' 'guava' 'hazelnurt' 'hazelnut' 'herb' 'honey' 'jam' 'jasmin' 'juici' 'lassi' 'lemon' 'linger'
    #  'liquroic' 'macadamia' 'mandarin' 'mango' 'mapl' 'marzepan' 'melon' 'milk' 'nectarin' 'nut' 'nutmeg' 'orang'
    #  'papaya' 'passion' 'pastri' 'peach' 'peanut' 'peanutbutt' 'pear' 'pineappl' 'pink' 'plum' 'pomelo' 'pralin'
    #  'raisin' 'raspberri' 'red' 'rhubarb' 'ripe' 'roast' 'rose' 'rum' 'smoke' 'spice' 'stick' 'stone' 'strawberri'
    #  'sugar' 'sweet' 'syrup' 'tea' 'tropic' 'vanilla' 'vibrant' 'violet' 'walnut' 'white' 'wild' 'yellow']
    # there are some mistakes (blosom, current, goosberry, hazelnurt) but they are inevitable

    return train_profiles_vector, test_profiles_vector
