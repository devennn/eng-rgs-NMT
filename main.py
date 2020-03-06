import os
import sys
from pathlib import Path

from preprocess import preprocess, split
from train import train, evaluate

if sys.argv[2] == 'local':
    # Load and preprocess dataset local
    abs_path = Path('.').parent.absolute()
    data_fname = os.path.join(abs_path, 'dataset', sys.argv[1])

elif sys.argv[2] == 'floydhub':
    # Load and preprocess dataset on floydhub
    abs_path = Path('.').parent.absolute()
    dir = os.listdir('/malay/')
    print(dir)
    data_fname = os.path.join('/malay', sys.argv[1])

print('Reading from: {}'.format(data_fname))


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
