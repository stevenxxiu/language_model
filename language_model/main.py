import datetime
import itertools
import random
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import clip_ops, init_ops


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


def load_data():
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

    # inputs
    X = tf.placeholder(tf.int32, [None, step_size])
    y = tf.placeholder(tf.int32, [None])
    lr = tf.placeholder(tf.float32, [])
    training = tf.placeholder(tf.bool, [])

    # network
    emb = tf.nn.embedding_lookup(tf.Variable(tf.random_normal(
        [len(word_to_index), embedding_size], stddev=0.01)
    ), X)
    if drop_out_apply in ('embedding', 'both'):
        emb = tf.layers.dropout(emb, drop_out, training=training)
    emb = tf.split(tf.reshape(tf.transpose(emb, [1, 0, 2]), [-1, embedding_size]), step_size, 0)
    lstm_cell = rnn.BasicLSTMCell(hidden_size)
    outputs, states = rnn.static_rnn(lstm_cell, emb, dtype=tf.float32)
    lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn')
    lstm = outputs[-1]
    if drop_out_apply in ('output', 'both'):
        lstm = tf.layers.dropout(lstm, drop_out, training=training)
    dense = tf.layers.dense(lstm, len(word_to_index), kernel_initializer=init_ops.glorot_uniform_initializer())
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=dense))

    # grad
    opt = None
    if optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif optimizer == 'adam':
        opt = tf.train.AdamOptimizer(initial_lr)
    elif optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(initial_lr)
    grads_and_vars = opt.compute_gradients(loss)
    for i, (grad, var) in enumerate(grads_and_vars):
        if var in lstm_vars:
            grads_and_vars[i] = (clip_ops.clip_by_value(grad, -100, 100), var)
    train_op = opt.apply_gradients(grads_and_vars)

    def all_loss(X_, y_):
        total_loss = 0
        for k in range(0, len(y_), batch_size):
            batch_X_, batch_y_ = X_[k:k + batch_size], y_[k:k + batch_size]
            total_loss += sess.run(loss, feed_dict={X: batch_X_, y: batch_y_, training: False}) * len(batch_y_)
        # geometric average of perplexity
        return np.exp(total_loss / len(y_))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prev_val_loss = np.inf
        for i in range(epoch_size):
            # generate minibatches
            p = np.random.permutation(len(train_y))

            # train
            train_X, train_y = train_X[p], train_y[p]
            for j in range(0, len(train_y), batch_size):
                if j % 256000 == 0:
                    # progress indicator
                    print(datetime.datetime.now(), j, all_loss(val_X, val_y))
                batch_X, batch_y = train_X[j:j + batch_size], train_y[j:j + batch_size]
                sess.run(train_op, feed_dict={
                    X: batch_X, y: batch_y, training: True, lr: initial_lr ** max(i + 1 - 4, 0)
                })

            # validate on epoch
            val_loss = all_loss(val_X, val_y)
            if early_stopping and val_loss >= prev_val_loss:
                break
            prev_val_loss = val_loss
        print(datetime.datetime.now(), 'final', all_loss(val_X, val_y), all_loss(test_X, test_y))


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
    train_words, val_words, test_words = load_data()
    while True:
        params = [random.choice(param_choices) for param_choices in params_choices]
        print(params)
        run_model(train_words, val_words, test_words, 512, *params)
        return

if __name__ == '__main__':
    main()
