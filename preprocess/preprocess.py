import pickle
from unicodedata import normalize
from pathlib import Path
import os

from .clean import clean_lines, spot_check

def load_file(filename):
	with open(filename, mode='rt', encoding='utf-8') as f:
		text = f.read()

	# split into sentences
	lines = text.strip().split('\n')
	pairs = [line.split('\t') for line in lines]
	return pairs

def save_clean_data(sentences, filename):
	path = Path('clean_dataset').parent.absolute()
	path = os.path.join(path, 'clean_dataset', filename)
	pickle.dump(sentences, open(path, 'wb'))
	print('Saved: %s' % filename)

def load_dataset(data_fname, save_fname):
	pairs = load_file(data_fname)
	clean_data = clean_lines(pairs)
	print(clean_data.shape)
	save_clean_data(clean_data, save_fname)
	# spot_check(clean_data)
