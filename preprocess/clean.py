import string
import numpy as np
import re
from nltk.corpus import stopwords
import sys


# clean a list of lines
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

def spot_check(clean_data):
	for i in range(len(clean_data)):
		print('[%s] => [%s]' % (clean_data[i,0], clean_data[i,1]))
