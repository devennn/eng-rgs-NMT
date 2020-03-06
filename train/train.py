import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import os
import sys
import gc

from .define_model import *
from .encoding import prepare_token, encode_output, encode_sequences
from .utils import load_clean_sentences

def run_fit(model, X_train, y_train, X_test, y_test):

	path = Path('.').parent.absolute()
	filename = os.path.join(path, 'result', 'model.h5')
	checkpoint = ModelCheckpoint(
		filename,
		monitor='val_loss',
		verbose=2,
		mode='min'
	)

	model.fit(
		X_train, y_train,
		epochs=10,
		batch_size=10,
	    validation_data=(X_test, y_test),
		callbacks=[checkpoint],
		verbose=2
	)

def run_train(full_path, train_path, test_path):
	# load datasets
	dataset = load_clean_sentences(full_path)
	train = load_clean_sentences(train_path)
	test = load_clean_sentences(test_path)

	eng_token, eng_len, trn_token, trn_len = prepare_token(dataset)

	# Vocab size of both datasets
	eng_vocab_size = len(eng_token.word_index) + 1
	trn_vocab_size = len(trn_token.word_index) + 1
	print('English Vocabulary Size: %d' % eng_vocab_size)
	print('TRANSLATED Vocabulary Size: %d' % trn_vocab_size)

	print('==== Prepare training Data ===')
	# prepare training data
	X_train = encode_sequences(trn_token, trn_len, train[:, 1])
	y_train = encode_sequences(eng_token, eng_len, train[:, 0])
	y_train = encode_output(y_train, eng_vocab_size)
	# prepare validation data
	X_test = encode_sequences(trn_token, trn_len, test[:, 1])
	y_test = encode_sequences(eng_token, eng_len, test[:, 0])
	y_test = encode_output(y_test, eng_vocab_size)

	# define model
	model = model_v1(
		source_vocab=trn_vocab_size,
		translate_vocab=eng_vocab_size,
		source_timesteps=trn_len,
		translate_timesteps=eng_len,
		n_units=256,
	)

	run_fit(model, X_train, y_train, X_test, y_test)
