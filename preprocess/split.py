import pickle
from numpy.random import rand
from numpy.random import shuffle
import sys
import os
from math import *
from pathlib import Path

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	pickle.dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# Create traing and test data
def create_train_test(raw_dataset):
	shuffle(raw_dataset)
	train_num = ceil(0.7 * raw_dataset.shape[0])
	test_num = raw_dataset.shape[0] - train_num
	train, test = raw_dataset[:train_num], raw_dataset[test_num:]
	return train, test

# split into train/test
def split(fname):
	# load dataset
	with open(fname, 'rb') as f:
		raw_dataset = pickle.load(f)
	train, test = create_train_test(raw_dataset)

	name = fname.split('.')
	path = Path('clean_dataset').parent.absolute()
	path = os.path.join(path, 'clean_dataset')
	full_name = name[0] + '_full' + '.pkl'
	full_name = os.path.join(path, full_name)
	train_name = name[0] + '_train' + '.pkl'
	train_name = os.path.join(path, train_name)
	test_name = name[0] + '_test' + '.pkl'
	test_name = os.path.join(path, test_name)

	save_clean_data(raw_dataset, full_name)
	save_clean_data(train, train_name)
	save_clean_data(test, test_name)
