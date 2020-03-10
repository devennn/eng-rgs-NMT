from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

# Disable Warnings
import os
os.environ['KMP_WARNINGS'] = 'off'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define NMT model
def model_v1(source_vocab, translate_vocab, source_timesteps,
    translate_timesteps, n_units, show_summary=False
):

    model = Sequential()
    model.add(Embedding(input_dim=source_vocab, output_dim=n_units,
        input_length=source_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(translate_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(translate_vocab, activation='softmax')))

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # summarize defined model
    if(show_summary == True):
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)

    return model
