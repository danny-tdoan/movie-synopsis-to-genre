import pickle
import numpy as np
import keras
import matplotlib.pyplot as plt

def load_tokenizer(tokenizer_loc):
    """load a pre-generated tokenizer"""

    with open(tokenizer_loc, 'rb') as handle:
        return pickle.load(handle)


def save_tokenizer(tokenizer_loc, tokenizer):
    with open(tokenizer_loc, 'wb') as handle:
        pickle.dump(tokenizer, handle)


def load_model(model_loc):
    """load a trained Keras model"""

    return keras.models.load_model(model_loc)


def save_model(model_loc, model):
    """Save a Keras model"""

    model.save(model_loc)


def predict_movie_genre(model, tokenized_syn, genres_np, to_plot=False, top_scores=3):
    """Given a plot synopsis, predict the genre"""
    syn_predicted = model.predict(tokenized_syn)

    # See the top predicted classification
    # get the top genres, in descending order of probs
    # Last `top_scores` is still ascending, hence reverse it.
    genres_txt = genres_np[np.ravel(syn_predicted).argsort()[-top_scores:][::-1]]
    probs = np.sort(np.ravel(syn_predicted))[::-1][:top_scores]
    if to_plot:
        plt.bar(np.arange(0, top_scores), probs, tick_label=genres_txt)

    return genres_txt, probs