import datetime
from collections import Counter

import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *
from nltk.corpus import brown


def words_to_mat(words, step_size, words_index):
    # length represents the index for an unknown word
    res = np.empty(len(words), dtype=int)
    for i, word in enumerate(words):
        res[i] = words_index.get(word, len(words_index))
    X = np.empty((len(words) - step_size, step_size), dtype=int)
    y = np.empty((len(words) - step_size,), dtype=int)
    for i in range(len(words) - step_size):
        X[i] = res[i: i + step_size]
        y[i] = res[i + step_size]
    return X, y


def run_model(
    batch_size, embedding_size, hidden_size, optimizer, initial_lr, step_size, epoch_size, drop_out,
    drop_out_apply, early_stopping, vocab_min_freq
):
    # convert data to index matrix
    words = brown.words()
    all_words = Counter(words)
    words_index = {}
    for word, count in all_words.items():
        if count >= vocab_min_freq:
            words_index[word] = len(words_index)
    train_X, train_y = words_to_mat(words[:800000], step_size, words_index)
    val_X, val_y = words_to_mat(words[800000:1000000], step_size, words_index)
    test_X, test_y = words_to_mat(words[1000000:], step_size, words_index)

    # network
    input_var = T.imatrix('input')
    l_in = lasagne.layers.InputLayer((None, step_size), input_var)
    l_emb = lasagne.layers.EmbeddingLayer(l_in, len(words_index) + 1, embedding_size)
    if drop_out_apply in ('embedding', 'both'):
        l_emb = lasagne.layers.DropoutLayer(l_emb, drop_out)
    l_forward = lasagne.layers.LSTMLayer(l_emb, hidden_size, grad_clipping=100, nonlinearity=tanh)
    if drop_out_apply in ('output', 'both'):
        l_forward = lasagne.layers.DropoutLayer(l_forward, drop_out)
    l_out = lasagne.layers.DenseLayer(l_forward, num_units=len(words_index) + 1, nonlinearity=softmax)

    # vars
    target_values = T.ivector('target_output')
    network_output = lasagne.layers.get_output(l_out)
    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.adagrad(cost, all_params, initial_lr)

    # functions
    train = theano.function([input_var, target_values], cost, updates=updates)
    compute_cost = theano.function([input_var, target_values], cost)

    def val_cost(X, y):
        total_cost = 0
        for k in range(0, len(val_y), batch_size):
            batch_X_, batch_y_ = X[k:k + batch_size], y[k:k + batch_size]
            total_cost += compute_cost(batch_X_, batch_y_) * len(batch_y_)
        # geometric average of perplexity
        return 2 ** (total_cost / len(y))

    for i in range(epoch_size):
        # generate minibatches
        p = np.random.permutation(len(train_y))
        train_X, train_y = train_X[p], train_y[p]
        for j in range(0, len(train_y), batch_size):
            if j % 1000 == 0:
                print(datetime.datetime.now(), j, val_cost(val_X, val_y), val_cost(test_X, test_y))
            batch_X, batch_y = train_X[j:j + batch_size], train_y[j:j + batch_size]
            train(batch_X, batch_y)
    print(datetime.datetime.now(), 'final', val_cost(val_X, val_y), val_cost(test_X, test_y))


def main():
    params = [
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
    run_model(512, *(param[0] for param in params))

if __name__ == '__main__':
    main()
