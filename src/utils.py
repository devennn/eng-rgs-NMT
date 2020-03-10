import pickle

def load_clean_sentences(filename):
	with open(filename, 'rb') as f:
		sentences = pickle.load(f)
	return sentences
