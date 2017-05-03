#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time
import chb
import matplotlib.pyplot as plt

import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne.objectives import binary_crossentropy, binary_accuracy
from sklearn import metrics

import networks as nw

# ##################### Build the neural network model #######################

def compile_model(input_var, target_var, net):

    prediction = layers.get_output(net['out'])
    loss = binary_crossentropy(prediction, target_var)
    loss = lasagne.objectives.aggregate(loss)

    params = layers.get_all_params(net['out'], trainable=True)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=1e-5)

    test_prediction = layers.get_output(net['out'], deterministic=True)
    test_loss = binary_crossentropy(test_prediction, target_var)
    test_loss = lasagne.objectives.aggregate(test_loss)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)
    prob_fn = theano.function([input_var], test_prediction)

    return train_fn, val_fn, prob_fn


# ##################### Testing function for ConvNet #######################

def nn_test(x_test, y_test, val_fn, prob_fn, batch_size=10):
    print('Test Results:')
    print('=' * 80)

    batch_err = []
    for batch in iterate_minibatches(x_test, y_test, batch_size):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        batch_err.append(err)

    test_err = np.mean(batch_err)

    print('Test loss: %.6f' % test_err)
    print('-' * 80)

    y_prob = prob_fn(x_test)
    y_pred = y_prob > 0.5
    print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))
    print('Matthews Correlation Coefficient:', metrics.matthews_corrcoef(y_test, y_pred))
    print('-' * 80)
    print(np.ravel(y_prob))
    print(np.ravel(y_pred).astype('int'))
    print(y_test)
    print('=' * 80)
    return test_err, y_pred, y_prob


# ############################# Batch iterator ###############################


def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################


def main(subject='chb05', num_epochs=10, tiger=False, plotter=False):
    # Load the dataset
    subj = chb.load_dataset(subject, tiger=tiger)
    sys.stdout.flush()
    batch_size = 10

    num_szr = subj.get_num()
    test_accs = [0] * num_szr
    out_dict = {}
    for szr in range(1, num_szr + 1):
        print('\nLeave-One-Out: %d of %d' % (szr, num_szr))

        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        net = nw.deep1(input_var)
        train_fn, val_fn, prob_fn = compile_model(input_var, target_var, net)

        train_err_list = [0] * num_epochs
        val_err_list   = [0] * num_epochs

        print('=' * 80)
        print('| epoch \t| train loss\t| val loss\t| time\t')
        print('=' * 80)

        x_test, y_test = 0, 0
        for epoch in range(num_epochs):
            st = time.clock()
            # make generator
            loo_gen = chb.lgus(subj, szr, batchsec=60, drop_prob=0.8, shuffle=True)
            # get test data (same on every epoch, so not really using it until the
            # last go-round)
            for batch in loo_gen:
                x_test, y_test = batch
                break
            # separate val and train data
            x_val = np.zeros((1000, 1, 23, 1280), dtype='float32')
            y_val = np.zeros((1000), dtype='int32')

            x_train, y_train = 0, 0
            batch_train_errs = []
            for idx, batch in enumerate(loo_gen):
                x_train, y_train = batch
                if idx < 1000:
                    x_train, x_val[idx] = x_train[:-1], x_train[-1:]
                    y_train, y_val[idx] = y_train[:-1], y_train[-1:]
                err = train_fn(x_train, y_train)
                batch_train_errs.append(err)
            epoch_train_err = np.mean(batch_train_errs)
            train_err_list[epoch] = epoch_train_err

            batch_val_errs = [0] * int(1000/batch_size)
            for idx, batch in enumerate(iterate_minibatches(x_val, y_val,
                                                            batch_size)):
                inputs, targets = batch
                err = val_fn(inputs, targets)
                batch_val_errs[idx] = err
            epoch_val_err = np.mean(batch_val_errs)
            val_err_list[epoch] = epoch_val_err

            en = time.clock()
            print('| %d \t\t| %.6f\t| %.6f\t| %.2f s' %
                  (epoch + 1, epoch_train_err, epoch_val_err, en - st))
        print('-' * 80)


        print('Training Complete.\n')
        if plotter:
            fig = plt.figure()
            plt.plot(range(num_epochs), train_err, label='Training error')
            plt.plot(range(num_epochs), val_err, label='Validation error')
            plt.title('ConvNet Training')
            plt.xlabel('Epochs')
            plt.ylabel('Error')
            plt.legend()
            plt.show()

        test_err, y_pred, y_prob = nn_test(x_test, y_test, val_fn, prob_fn, batch_size)
        out_dict['_'.join(['prob', str(szr)])] = y_prob
        out_dict['_'.join(['true', str(szr)])] = y_test

    np.savez(''.join([subject, 'deep.npz']), **out_dict)

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['subject'] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs['num_epochs'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        kwargs['tiger'] = bool(sys.argv[3])
    if len(sys.argv) > 4:
        kwargs['plotter'] = bool(sys.argv[4])
    main(**kwargs)
