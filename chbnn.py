#!/usr/bin/env python
"""
First attempt at Lasagne with my real data.
"""

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
from lasagne.nonlinearities import rectify, leaky_rectify, sigmoid
from lasagne.objectives import binary_crossentropy, binary_accuracy

# ##################### Build the neural network model #######################


def scratch_net(input_var, data_size=(None, 1, 23, 256), output_size=1):
    net = {}
    net['data'] = layers.InputLayer(data_size, input_var=input_var)
    net['conv1'] = layers.Conv2DLayer(
        net['data'],
        num_filters=4,
        filter_size=(1, 7),
        pad='same',
        nonlinearity=rectify)
    net['conv2'] = layers.Conv2DLayer(
        net['conv1'],
        num_filters=8,
        filter_size=(1, 15),
        pad='same',
        stride=(1, 2),
        nonlinearity=rectify)
    net['pool'] = layers.MaxPool2DLayer(net['conv2'], pool_size=(1, 2))
    net['fcl'] = layers.DenseLayer(
        net['pool'], num_units=256, nonlinearity=rectify)
    net['out'] = layers.DenseLayer(
        net['fcl'], num_units=output_size, nonlinearity=sigmoid)
    return net


def scratch_model(input_var, target_var, net):

    prediction = layers.get_output(net['out'])
    loss = binary_crossentropy(prediction, target_var)
    loss = lasagne.objectives.aggregate(loss)

    params = layers.get_all_params(net['out'], trainable=True)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=1e-5)

    test_prediction = layers.get_output(net['out'], deterministic=True)

    test_loss = binary_crossentropy(test_prediction, target_var)
    test_loss = lasagne.objectives.aggregate(test_loss)
    test_acc = T.mean(
        binary_accuracy(test_prediction, target_var),
        dtype=theano.config.floatX)

    train_fn = theano.function(
        [input_var, target_var],
        loss,
        updates=updates,
        allow_input_downcast=True)
    val_fn = theano.function(
        [input_var, target_var], [test_loss, test_acc],
        allow_input_downcast=True)

    return train_fn, val_fn


def scratch_train(train_fn, val_fn, num_epochs):
    train_err_list = []
    val_err_list = []
    val_acc_list = []

    print('=' * 80)
    print('| epoch \t| train loss\t| val loss\t| val acc\t| time\t')
    print('=' * 80)

    for epoch in range(num_epochs):
        st = time.time()
        batch_train_errs = []
        for batch in iterate_minibatches(x_train, y_train, batch_size):
            inputs, targets = batch
            err = train_fn(inputs, targets)
            batch_train_errs.append(err)
        epoch_train_err = np.mean(batch_train_errs)
        train_err_list.append(epoch_train_err)

        batch_val_errs = []
        batch_val_accs = []
        for batch in iterate_minibatches(x_val, y_val, batch_size):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            batch_val_errs.append(err)
            batch_val_accs.append(acc)
        epoch_val_err = np.mean(batch_val_errs)
        val_err_list.append(epoch_val_err)
        epoch_val_acc = np.mean(batch_val_accs)
        val_acc_list.append(epoch_val_acc)

        en = time.time()
        print('| %d \t\t| %.6f\t| %.6f\t| %.2f%%\t| %.2f s' %
              (epoch + 1, epoch_train_err, epoch_val_err, epoch_val_acc * 100,
               en - st))
    print('-' * 80)
    return train_err_list, val_err_list, val_acc_list


def scratch_test(val_fn):
    print('Test Results:')
    print('=' * 80)

    batch_err = []
    batch_acc = []
    for batch in iterate_minibatches(x_test, y_test, batch_size):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        batch_err.append(err)
        batch_acc.append(acc)

    test_err = np.mean(batch_err)
    test_acc = np.mean(batch_acc)

    print('Test loss: %.6f' % test_err)
    print('Test accuracy: %.2f' % (test_acc * 100))
    print('-' * 80)
    return test_err, test_acc


# ############################# Batch iterator ###############################


def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################

if len(sys.argv) > 1:
    subject = sys.argv[1]
else:
    subject = 'chb01'
if len(sys.argv) > 2:
    num_epochs = int(sys.argv[2])
else:
    num_epochs = 50
if len(sys.argv) > 3:
    tiger = bool(sys.argv[3])
else:
    tiger = False
if len(sys.argv) > 4:
    plotter = bool(sys.argv[4])
else:
    plotter = False

# Load the dataset
subj = chb.load_dataset(subject, tiger=tiger)
sys.stdout.flush()

num = subj.get_num()
test_accs = []
for szr in range(1, num + 1):
    print('\nLeave-One-Out: %d of %d' % (szr, num))
    x_train, y_train, x_test, y_test = chb.leaveOneOut(subj, szr)
    x_train, x_val = x_train[:-100], x_train[-100:]
    y_train, y_val = y_train[:-100], y_train[-100:]

    batch_size = 10

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    net = scratch_net(input_var)
    train_fn, val_fn = scratch_model(input_var, target_var, net)

    train_err, val_err, val_acc = scratch_train(train_fn, val_fn, num_epochs)
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

        fig2 = plt.figure()
        plt.plot(range(num_epochs), np.asarray(val_acc) * 100)
        plt.title('ConvNet Training: Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

    test_err, test_acc = scratch_test(val_fn)
    test_accs.append(test_acc)
    sys.stdout.flush()

print('*-' * 40)
print('Average test accuracy for %d Leave-One-Out tests: %.2f' %
      (num, np.mean(test_accs) * 100))
print('*-' * 40)
print()
# Optionally, you could now dump the network weights to a file like this:
# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)
