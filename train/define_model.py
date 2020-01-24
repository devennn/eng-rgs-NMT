from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model

# Disable Warnings
import os
os.environ['KMP_WARNINGS'] = 'off'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps,
    n_units, show_summary=False):

    model = Sequential([
        Embedding(src_vocab, n_units, input_length=src_timesteps,
            mask_zero=True),
        LSTM(n_units),
        RepeatVector(tar_timesteps),
        LSTM(n_units, return_sequences=True),
        TimeDistributed(Dense(tar_vocab, activation='softmax'))
    ])

    # Compile the model
    model.compile(
        # Loss function
        loss='categorical_crossentropy',
        # Optimizer function
        optimizer='adam',
        metrics=['accuracy']
    )

    # summarize defined model
    if(show_summary == True):
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)

    return model