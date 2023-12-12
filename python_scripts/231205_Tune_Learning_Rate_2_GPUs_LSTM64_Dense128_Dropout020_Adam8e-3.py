from datetime import datetime
import sys # logging
import os # basename

# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Open a text file for writing output of all print statements and stderr
base_name = os.path.splitext(os.path.basename(__file__))[0] # name of log file should include name of this script
log_file_name = f"logs/{base_name}_{current_time}.log"
log_file = open(log_file_name, 'w')

# Redirect both standard output and standard error to the log file
sys.stdout = log_file
sys.stderr = log_file

#######################################################################################################################################

# imports
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, Bidirectional

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import gc

print(tf.config.list_physical_devices())

gc.collect()

# Load data from the npz file
data = np.load('../231117_encoded_seqs_labels_int8.npz')
sequences = data['encoded_sequences']
labels = data['input_labels']

# cleanup
tf.keras.backend.clear_session()
gc.collect()

# uncompiled model
def create_uncompiled_model():
    #tf.keras.backend.clear_session()
    gc.collect()
    
    tf.random.set_seed(42)

    # create model
    model = Sequential([
        Input(shape=(201, 4)), # see sequences.shape
        Bidirectional(LSTM(64)), 
        Dense(128, activation='relu'), 
        Dropout(0.2), 
        Dense(2, activation='softmax')
    ])

    model.summary()

    return model

# set up for multiGPU
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

with strategy.scope():
    # Create & compile model
    model = create_uncompiled_model()

    lr_schedule = LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 18)) # 72 epochs spans 1e-5 to 1e-1; 72/18 = 4

    # compile model
    model.compile(optimizer=Adam(), 
                 loss='binary_crossentropy', 
                 metrics='accuracy')

    model.summary()

# Fit the model
epochs = 72
batch_size = 2**12
history = model.fit(sequences, labels, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    callbacks=[lr_schedule])

# save objects to pickle
pkl_file_name = f"pkls/{base_name}_{current_time}.pkl"
with open(pkl_file_name, 'wb') as file:
    pickle.dump(history, file)

#######################################################################################################################################

# close the log file
log_file.close()
