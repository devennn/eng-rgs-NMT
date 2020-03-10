import pickle
from unicodedata import normalize
from pathlib import Path
import string
import numpy as np
import re
from nltk.corpus import stopwords
from numpy.random import rand
from numpy.random import shuffle
from math import *
import os

def load_dataset(data_fname, save_fname):

	with open(data_fname, mode='rt', encoding='utf-8') as f:
		text = f.read()

	# split into sentences
	lines = text.strip().split('\n')
	pairs = [line.split('\t') for line in lines]

	clean_data = clean_lines(pairs)
	print(clean_data.shape)
	save_clean_data(clean_data, save_fname)

def save_clean_data(sentences, filename):
	path = Path('clean_dataset').parent.absolute()
	path = os.path.join(path, 'clean_dataset', filename)
	with open(path, 'wb+') as f:
		pickle.dump(sentences, f)
	print('Saved: %s' % filename)

def clean_lines(lines):
	cleaned = []
	top_rgs = ['do', 'ong', 'no', 'oku', 'ku', 'sid', 'di',
				'ko', 'nu', 'tu', 'po', 'dino', 'diti', 'iti',
				'dot', 'i', 'ka', 'ilo', 'ino', 'o']
	top_en = ['of', 'a', 'the', 'to', 'is', 'i', 'you', 'in',
			'take']
	table = str.maketrans("", "", string.punctuation)
	for pair in lines:
		clean_pair = []
		for line in pair:
			# tokenize on white space
			line = line.split()
			# remove punctuation
			line = [s.translate(table) for s in line]
			# convert to lowercase
			line = [word.lower() for word in line]
			# Remove top frequent words
			for word in line:
				if word in top_rgs or word in top_en:
					line.remove(word)
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)

	return np.array(cleaned)

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
	pickle.dump(raw_dataset, open(full_name, 'wb'))
	print('Saved: %s' % full_name)

	train_name = name[0] + '_train' + '.pkl'
	train_name = os.path.join(path, train_name)
	pickle.dump(train, open(train_name, 'wb'))
	print('Saved: %s' % train_name)

	test_name = name[0] + '_test' + '.pkl'
	test_name = os.path.join(path, test_name)
	pickle.dump(test, open(test_name, 'wb'))
	print('Saved: %s' % test_name)
