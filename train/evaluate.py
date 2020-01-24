from pickle import load
import numpy as np
from pathlib import Path
import os
import ast
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from .utils import load_clean_sentences
from .encoding import create_token, max_len, encode_sequences, word_for_id
from .accuracy import evaluate_accuracy

def prepare_tokenizer(dataset, train, test):
    # prepare english tokenizer
    eng_tokenizer = create_token(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_len(dataset[:, 0])

    # prepare dsn tokenizer
    dsn_tokenizer = create_token(dataset[:, 1])
    dsn_vocab_size = len(dsn_tokenizer.word_index) + 1
    dsn_length = max_len(dataset[:, 1])

    # prepare data
    X_train = encode_sequences(dsn_tokenizer, dsn_length, train[:, 1])
    X_test = encode_sequences(dsn_tokenizer, dsn_length, test[:, 1])
    return X_train, X_test, eng_tokenizer

def run_evaluate(full_path, train_path, test_path):

    # load datasets
    dataset = load_clean_sentences(full_path)
    train = load_clean_sentences(train_path)
    test = load_clean_sentences(test_path)

    X_train, X_test, eng_tokenizer = prepare_tokenizer(dataset, train, test)

    # load model
    path = Path('.').parent.absolute()
    file_path = os.path.join(path, 'result', 'model.h5')
    model = load_model(file_path)
    # test on some training sequences
    print('train')
    evaluate_accuracy(model, eng_tokenizer, X_train, train)
    # test on some test sequences
    print('test')
    evaluate_accuracy(model, eng_tokenizer, X_test, test)
