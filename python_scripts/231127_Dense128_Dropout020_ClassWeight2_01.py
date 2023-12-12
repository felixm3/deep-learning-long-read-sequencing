from datetime import datetime
import sys # logging
import os # basename

# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Open a text file for writing output of all print statements and stderr
base_name = os.path.splitext(os.path.basename(__file__))[0]
log_file_name = f"logs/{base_name}_{current_time}.log"
log_file = open(log_file_name, 'w')

# Redirect both standard output and standard error to the log file
sys.stdout = log_file
sys.stderr = log_file

#######################################################################################################################################

import numpy as np
import tensorflow as tf
import gc
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

gc.collect()

# Load data from the npz file
data = np.load('../231117_encoded_seqs_labels_int8.npz')
sequences = data['encoded_sequences']
labels = data['input_labels']

# cleanup
tf.keras.backend.clear_session()
gc.collect()

# create model
model = Sequential([
    Input(shape=(201, 4)), # see sequences.shape
    Flatten(), 
    Dense(128, activation='relu'), 
    Dropout(0.2), 
    Dense(2, activation='softmax')
])

# Compile the model with appropriate loss and optimizer
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),  # tunable hyperparameter: learning rate
              metrics=['accuracy'])

model.summary()

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    min_delta=0.0001,         # Minimum change to be considered an improvement
    patience=100,              # Number of epochs with no improvement before stopping
    mode='max',               # We want to maximize validation accuracy
    verbose=1                 # Print a message when training stops
)

# Fit the model
epochs = 1000
history = model.fit(sequences, labels, 
                    epochs=epochs, 
                    batch_size=2**12, 
                    class_weight={0: 1, 1: 2}, # class weights
                    callbacks=[early_stopping],  # Include the EarlyStopping callback
                    validation_split=0.2)  # tunable hyperparameters: batch size and epochs


# save objects to pickle
pkl_file_name = f"pkls/{base_name}_{current_time}.pkl"
with open(pkl_file_name, 'wb') as file:
    pickle.dump(history, file)

#######################################################################################################################################

# close the log file
log_file.close()
