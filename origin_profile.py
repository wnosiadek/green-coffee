"""
Script for classifying green coffee origin based on its sensory profile

Transforms the data using 'origins.py', vectorizes the sensory profiles using Count Vectorizer with Nltk library
(tokenization and stemming), creates Multinomial and Complement Naive Bayes Classifiers and fits them on the training
set, tests the models on the testing set

Requires installation of 'pandas', 'nltk', 'scikit-learn'
"""

import pandas
import adjust_data
import origins
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB

print('\nClassifying green coffee origin based on its sensory profile...')

# read relevant training data from the adjusted file
data = pandas.read_csv('adjusted_data.csv')
train_profiles = data['Profile']
train_origins = data['Coffee']
# drop rows with missing values using 'adjust_data.py'
train_profiles, train_origins = adjust_data.drop_missing(train_profiles, train_origins)

# read relevant testing data from the adjusted file
test_data = pandas.read_csv('adjusted_test_data.csv')
test_profiles = test_data['Profile']
test_origins = test_data['Coffee']
# drop rows with missing values using 'adjust_data.py'
test_profiles, test_origins = adjust_data.drop_missing(test_profiles, test_origins)

# transform training and testing data using 'origins.py'
# (simplify possible origins, take into account continents rather than countries, change alphabetical data into
#  numerical data)
train_profiles, train_origins = origins.simplify_origins(train_profiles, train_origins)
test_profiles, test_origins = origins.simplify_origins(test_profiles, test_origins)
# print the mapping for reference
print(f'\n{origins.printable_origins_map}')


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
    # remove stopwords and punctuation using check
    stemmer = SnowballStemmer('english')
    text_tokens = [stemmer.stem(token) for token in text_tokens if check(token)]

    return text_tokens


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
#  'eleg' 'fig' 'floral' 'flower' 'floweri' 'fruit' 'fudg' 'goosberri' 'gooseberri' 'grape' 'grapefruit' 'grass' 'grey'
#  'grill' 'guava' 'hazelnurt' 'hazelnut' 'herb' 'honey' 'jam' 'jasmin' 'juici' 'lassi' 'lemon' 'linger' 'liquroic'
#  'macadamia' 'mandarin' 'mango' 'mapl' 'marzepan' 'melon' 'milk' 'nectarin' 'nut' 'nutmeg' 'orang' 'papaya' 'passion'
#  'pastri' 'peach' 'peanut' 'peanutbutt' 'pear' 'pineappl' 'pink' 'plum' 'pomelo' 'pralin' 'raisin' 'raspberri' 'red'
#  'rhubarb' 'ripe' 'roast' 'rose' 'rum' 'smoke' 'spice' 'stick' 'stone' 'strawberri' 'sugar' 'sweet' 'syrup' 'tea'
#  'tropic' 'vanilla' 'vibrant' 'violet' 'walnut' 'white' 'wild' 'yellow']
# there are some mistakes (blosom, current, goosberry, hazelnurt) but they are inevitable

# create Multinomial and Complement Naive Bayes Classifiers for the training data
multinomial_classifier = MultinomialNB()
multinomial_classifier.fit(train_profiles_vector, train_origins.to_numpy())
complement_classifier = ComplementNB()
complement_classifier.fit(train_profiles_vector, train_origins.to_numpy())

# make predictions on the testing data
multinomial_predicted_origins = multinomial_classifier.predict(test_profiles_vector)
multinomial_accuracy = multinomial_classifier.score(test_profiles_vector, test_origins.to_numpy())
complement_predicted_origins = complement_classifier.predict(test_profiles_vector)
complement_accuracy = complement_classifier.score(test_profiles_vector, test_origins.to_numpy())

# the results
print(f'\nTrue origins: {test_origins.to_numpy()}')
print(f'Origins predicted with Multinomial Naive Bayes: {multinomial_predicted_origins}')
print(f'Accuracy (mean accuracy): {multinomial_accuracy:.2f}')
print(f'\nTrue origins: {test_origins.to_numpy()}')
print(f'Origins predicted with Complement Naive Bayes: {complement_predicted_origins}')
print(f'Accuracy (mean accuracy): {complement_accuracy:.2f}')
