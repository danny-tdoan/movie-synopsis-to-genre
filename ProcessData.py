import logging
import pandas as pd
import numpy as np
import Utilities as util
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from constants import VOCAB_SIZE

def get_all_genres(data):
    """Aggregate all the possible genres that appear in the data"""

    total_genres = []
    for movie in data['genres'].values:
        total_genres.extend(movie)

    return list(set(total_genres))


def get_top_n_genres(data, n):
    """Shortlist the most common genres out of excessive number of total genres"""
    total_genres = []
    for movie in data['genres'].values:
        total_genres.extend(movie)

    genre_tally = pd.Series(total_genres).value_counts()

    return genre_tally.nlargest(n).index.tolist()


def categorize_movie_genre(data, genres):
    """one hot encoder or get_dummies don't really work
        there are multiple labels in non consistent order. Manually categorize the genre"""
    labels = (np.in1d(genres, data['genres'].values[0]))

    for i in range(1, data['genres'].shape[0]):
        labels = np.vstack((labels, np.in1d(genres, data['genres'].values[i])))

    labels = labels.astype(int)

    return labels


def prepare_train_test_set(X, y, tokenizer=None, save_tokenizer=False, vocab_size=VOCAB_SIZE, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # generate the tokenizer if necessary
    if tokenizer == None:
        logging.info("Tokenizer is null, initialize it")
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(X_train)

    # assume the file to saved tokenizer is provided
    elif type(tokenizer == str):
        logging.info("Tokenizer is loaded from {}".format(tokenizer))
        tokenizer = util.load_tokenizer(tokenizer)

    if save_tokenizer:
        logging.info("Saving tokenizer to tokenizer/tokenizer")
        util.save_tokenizer('tokenizer/tokenizer', tokenizer)

    X_train_tokenized = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    logging.info("Finish tokenizing X_train")

    X_test_tokenized = tokenizer.texts_to_matrix(X_test, mode='tfidf')
    logging.info("Finish tokenizing X_test")

    return (X_train_tokenized, y_train), (X_test_tokenized, y_test)
