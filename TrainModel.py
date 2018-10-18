import os
import logging
import LoadData as data_load
import ProcessData as data_process
import Utilities as utils
from keras import Sequential
from keras.layers import Activation, Dense, Dropout
from constants import *

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG, filename=os.path.join(LOG_DIR, LOG_NAME), filemode='w')

data = data_load.load_data(SYNOPSIS_FILE, MOVIE_META_FILE)
logging.info("Data loaded")

# get all genres and categorize the labels
top_genres = data_process.get_top_n_genres(data, TOP_N_GENRES)
labels = data_process.categorize_movie_genre(data, top_genres)
logging.info("Shortlisted {} genres and encoded labels".format(TOP_N_GENRES))

train, test = data_process.prepare_train_test_set(data['text'], labels, save_tokenizer=True)
logging.info("Splitted train and test set")

# build the model on train data
logging.info("Building model")
model_multi = Sequential()
# depth of 1
model_multi.add(Dense(512, input_shape=(VOCAB_SIZE,)))
model_multi.add(Activation('relu'))

# drop some nodes for regularization here
model_multi.add(Dropout(0.25))

model_multi.add(Dense(512))
model_multi.add(Activation('relu'))
model_multi.add(Dropout(0.25))

# 50 genres in total
model_multi.add(Dense(len(top_genres)))

# use sigmoid at the last layer activation, and binary_crossentropy
# for multi label classification
# https://towardsdatascience.com/multi-label-classification-and-class-activation-map-on-fashion-mnist-1454f09f5925
model_multi.add(Activation('sigmoid'))
model_multi.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# introduce validation split to avoid overfitting
logging.info("Start training model")
try:
    out = model_multi.fit(train[0], train[1], epochs=5, verbose=MODEL_VERBOSE, validation_split=0.25)
except Exception as e:
    logging.error("Exception: ", exc_info=True)

model_name = 'model_multi.h5'
model_path = os.path.join(MODEL_DIR, model_name)
logging.info("Finished training. Saving model to {}".format(model_path))
utils.save_model(model_path, model_multi)
