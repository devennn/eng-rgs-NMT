import os
import sys
from pathlib import Path

from src import *

def get_dir(in_arg):

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

    return abs_path, data_fname

def get_path(fname, abs_path):

    name1 = '{}_full.pkl'.format(fname[0])
    name2 = '{}_train.pkl'.format(fname[0])
    name3 = '{}_test.pkl'.format(fname[0])
    full_path = os.path.join(abs_path, 'clean_dataset', name1)
    train_path = os.path.join(abs_path, 'clean_dataset', name2)
    test_path = os.path.join(abs_path, 'clean_dataset', name3)

    return full_path, train_path, test_path

if __name__ == '__main__':

    abs_path, data_fname = get_dir(sys.argv[2])
    fname = sys.argv[1].split('.')
    save_fname = fname[0] + '.pkl'
    load_dataset(data_fname, save_fname)

    # Split
    path = os.path.join(abs_path, 'clean_dataset', save_fname)
    split(path)

    # train
    full_path, train_path, test_path = get_path(fname, abs_path)
    run_train(full_path, train_path, test_path)

    # Evaluate
    run_evaluate(full_path, train_path, test_path)
