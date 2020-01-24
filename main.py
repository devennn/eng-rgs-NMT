import os
import sys
from pathlib import Path

from preprocess import preprocess, split
from train import train, evaluate

# Load and preprocess dataset
abs_path = Path('dataset').parent.absolute()
data_fname = os.path.join(abs_path, 'dataset', sys.argv[1])
fname = sys.argv[1].split('.')
save_fname = fname[0] + '.pkl'
preprocess.load_dataset(data_fname, save_fname)

# Split
path = os.path.join(abs_path, 'clean_dataset', save_fname)
split.split(path)

# train
name1 = '{}_full.pkl'.format(fname[0])
name2 = '{}_train.pkl'.format(fname[0])
name3 = '{}_test.pkl'.format(fname[0])
full_path = os.path.join(abs_path, 'clean_dataset', name1)
train_path = os.path.join(abs_path, 'clean_dataset', name2)
test_path = os.path.join(abs_path, 'clean_dataset', name3)
train.run_train(full_path, train_path, test_path)

# Evaluate
evaluate.run_evaluate(full_path, train_path, test_path)
