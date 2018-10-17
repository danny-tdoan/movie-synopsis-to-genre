import pandas as pd
import numpy as np

def get_all_genres(data):
    """Aggregate all the possible genres that appear in the data"""

    total_genres=[]
    for movie in data['genres'].values:
        total_genres.extend(movie)

    return list(set(total_genres))

def get_top_n_genres(data,n):
    total_genres=[]
    for movie in data['genres'].values:
        total_genres.extend(movie)

    genre_tally=pd.Series(total_genres).value_counts()

    return genre_tally.nlargest(n).index.tolist()

def categorize_movie_genre(data,all_genres):
    """one hot encoder or get_dummies don't really work
        there are multiple labels in non consistent order. Manually categorize the genre"""
    labels = (np.in1d(all_genres, data['genres'].values[0]))

    for i in range(1, data['genres'].shape[0]):
        labels = np.vstack((labels, np.in1d(all_genres, data['genres'].values[i])))

    labels = labels.astype(int)

    return labels.shape