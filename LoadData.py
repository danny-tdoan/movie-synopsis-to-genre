import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from constants import *

matplotlib.rcParams['figure.figsize'] = (15.0, 7.5)


def load_synopsis(file, columns=['id', 'text'], delim='\t'):
    """Load synopsis from file. If columns param is provided, it indicates the file originally has no column"""
    if not columns == None:
        synopsis = pd.read_csv(file, header=None, delimiter=delim)
        synopsis.columns = columns
    else:
        synopsis = pd.read_csv('data/plot_summaries.txt', delimiter=delim)

    return synopsis


def load_movie_genre(file, delim='\t'):
    """Load movie genre from file. only take the id and the genre. Store the genres as a list"""
    movies_meta = pd.read_csv(file, header=None, delimiter=delim)
    movies_genre = movies_meta[[META_COL_ID, META_COL_GENRE]]
    movies_genre.columns = ['id', 'genres']

    # update the genre column to list type
    movies_genre['genres'] = movies_genre['genres'].apply(eval).apply(lambda x: list(x.values()))

    return movies_genre


def load_data(syn_file=SYNOPSIS_FILE, meta_file=MOVIE_META_FILE, dropna=True):
    """Join synopsis and genre based on movie ID"""
    synopsis = load_synopsis(syn_file, columns=['id', 'text'])
    genre = load_movie_genre(meta_file)

    data = synopsis.set_index('id').join(genre.set_index('id'))
    if dropna:
        data = data.dropna()

    return data
