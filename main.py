import datetime
import itertools
import random
from collections import Counter

import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *


def words_to_mat(words, step_size, words_index):
    # length represents the index for an unknown word
    res = np.empty(len(words), dtype=np.int32)
    for i, word in enumerate(words):
        res[i] = words_index.get(word, words_index['<unk>'])
    X = np.empty((len(words) - step_size, step_size), dtype=np.int32)
    y = np.empty((len(words) - step_size,), dtype=np.int32)
    for i in range(len(words) - step_size):
        X[i] = res[i: i + step_size]
        y[i] = res[i + step_size]
    return X, y


def preprocess_data():
    # convert to one long sentence
    with open('../data/penn_treebank/ptb.train.txt', encoding='utf-8') as train_sr, \
            open('../data/penn_treebank/ptb.valid.txt', encoding='utf-8') as val_sr, \
            open('../data/penn_treebank/ptb.test.txt', encoding='utf-8') as test_sr:
        train_words = [word for sent in train_sr for word in sent.split() + ['</eos>']]
        val_words = [word for sent in val_sr for word in sent.split() + ['</eos>']]
        test_words = [word for sent in test_sr for word in sent.split() + ['</eos>']]
    return train_words, val_words, test_words


def run_model(
    train_words, val_words, test_words,
    batch_size, embedding_size, hidden_size, optimizer, initial_lr, step_size, epoch_size, drop_out,
    drop_out_apply, early_stopping, vocab_min_freq
):
    # convert data to index matrix
    word_to_freq = Counter(itertools.chain(train_words, val_words, test_words))
    word_to_index = {}
    for word, count in word_to_freq.items():
        if count >= vocab_min_freq:
            word_to_index[word] = len(word_to_index)
    train_X, train_y = words_to_mat(train_words, step_size, word_to_index)
    val_X, val_y = words_to_mat(val_words, step_size, word_to_index)
    test_X, test_y = words_to_mat(test_words, step_size, word_to_index)

    # network
    input_var = T.imatrix('input')
    l_in = lasagne.layers.InputLayer((None, step_size), input_var)
    l_emb = lasagne.layers.EmbeddingLayer(l_in, len(word_to_index) + 1, embedding_size)
    if drop_out_apply in ('embedding', 'both'):
        l_emb = lasagne.layers.DropoutLayer(l_emb, drop_out)
    l_forward = lasagne.layers.LSTMLayer(l_emb, hidden_size, grad_clipping=100, nonlinearity=tanh)
    if drop_out_apply in ('output', 'both'):
        l_forward = lasagne.layers.DropoutLayer(l_forward, drop_out)
    l_out = lasagne.layers.DenseLayer(l_forward, num_units=len(word_to_index) + 1, nonlinearity=softmax)

    # vars
    lr = T.scalar('lr')
    target_values = T.ivector('target_output')
    network_output = lasagne.layers.get_output(l_out)
    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = None
    if optimizer == 'sgd':
        updates = lasagne.updates.sgd(cost, all_params, lr)
    elif optimizer == 'adam':
        updates = lasagne.updates.adam(cost, all_params, initial_lr)
    elif optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(cost, all_params, initial_lr)

    # functions
    train = theano.function([input_var, target_values, lr], cost, updates=updates, on_unused_input='ignore')
    compute_cost = theano.function([input_var, target_values], cost)

    def all_cost(X, y):
        total_cost = 0
        for k in range(0, len(y), batch_size):
            batch_X_, batch_y_ = X[k:k + batch_size], y[k:k + batch_size]
            total_cost += compute_cost(batch_X_, batch_y_) * len(batch_y_)
        # geometric average of perplexity
        return np.exp(total_cost / len(y))

    prev_val_cost = np.inf
    for i in range(epoch_size):
        # generate minibatches
        p = np.random.permutation(len(train_y))

        # train
        train_X, train_y = train_X[p], train_y[p]
        for j in range(0, len(train_y), batch_size):
            if j % 256000 == 0:
                # progress indicator
                print(datetime.datetime.now(), j, all_cost(val_X, val_y))
            batch_X, batch_y = train_X[j:j + batch_size], train_y[j:j + batch_size]
            train(batch_X, batch_y, initial_lr ** max(i + 1 - 4, 0.0))

        # validate on epoch
        val_cost = all_cost(val_X, val_y)
        if early_stopping and val_cost >= prev_val_cost:
            break
        prev_val_cost = val_cost
    print(datetime.datetime.now(), 'final', all_cost(val_X, val_y), all_cost(test_X, test_y))


def main():
    params_choices = [
        (50, 100, 200, 300),
        (50, 100, 200, 300),
        ('sgd', 'adam', 'rmsprop'),
        (0.1, 0.01, 0.001),
        (5, 10, 20, 30),
        (5, 10, 20, 30),
        (0, 0.2, 0.4, 0.6, 0.8),
        ('output', 'embedding', 'both'),
        (True, False),
        (0, 5, 10, 15, 20),
    ]
    train_words, val_words, test_words = preprocess_data()
    while True:
        params = [random.choice(param_choices) for param_choices in params_choices]
        print(params)
        run_model(train_words, val_words, test_words, 512, *params)

if __name__ == '__main__':
    main()
