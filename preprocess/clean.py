import string
import numpy as np

# clean a list of lines
def clean_lines(lines):
	cleaned = []
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
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return np.array(cleaned)

def spot_check(clean_data):
	for i in range(len(clean_data)):
		print('[%s] => [%s]' % (clean_data[i,0], clean_data[i,1]))
