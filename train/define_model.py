from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Embedding, RepeatVector,
                                        TimeDistributed)
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

def model_v2(source_vocab, translate_vocab, latent_dim=256, show_summary=False):

    # Encoder
    encoder_inputs = Input(shape=(None, source_vocab))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = Input(shape=(None, translate_vocab))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(translate_vocab, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

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
