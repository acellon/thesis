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
from lasagne.nonlinearities import rectify, leaky_rectify, softmax
from lasagne.objectives import binary_crossentropy, binary_accuracy

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
import nolearn.lasagne

from sklearn.metrics import classification_report, accuracy_score

# ##################### Build the neural network model #######################

def make_net(input_var, data_size=(None, 1, 23, 1280), output_size=1):
    net = {}
    net['data'] = layers.InputLayer(data_size, input_var=input_var)
    net['conv1'] = layers.Conv2DLayer(
        net['data'],
        num_filters=8,
        filter_size=(1, 255),
        stride=(1, 32),
        pad='same',
        nonlinearity=rectify)
    net['conv2'] = layers.Conv2DLayer(
        net['conv1'],
        num_filters=8,
        filter_size=(1, 127),
        pad='same',
        stride=(1, 32),
        nonlinearity=rectify)
    net['pool'] = layers.MaxPool2DLayer(net['conv2'], pool_size=(1, 2))
    net['fcl'] = layers.DenseLayer(
        net['pool'], num_units=256, nonlinearity=rectify)
    net['out'] = layers.DenseLayer(
        net['fcl'], num_units=output_size, nonlinearity=sigmoid)
    return net


def compile_model(input_var, target_var, net):

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

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn

def nn_test(x_test, y_test, val_fn):
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
batch_size = 10

num_szr = subj.get_num()
test_accs = [0] * num_szr
for szr in range(1, num_szr + 1):
    print('\nLeave-One-Out: %d of %d' % (szr, num_szr))

    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool', layers.MaxPool2DLayer),
                ('fcl', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, 23, 1280),
        # layer conv1
        conv1_num_filters=8,
        conv1_filter_size=(1, 255),
        conv1_stride=(1, 32),
        conv1_pad='same',
        conv1_nonlinearity=rectify,
        # layer conv2
        conv2_num_filters=8,
        conv2_filter_size=(1, 127),
        conv2_pad='same',
        conv2_stride=(1, 32),
        conv2_nonlinearity=rectify,
        # layer pool
        pool_pool_size=(1,2),
        # layer fcl
        fcl_num_units=256,
        fcl_nonlinearity=rectify,
        # layer output
        output_num_units=2,
        output_nonlinearity=softmax,
        # optimization method params
        update=lasagne.updates.rmsprop,
        update_learning_rate=1e-5,
        objective_loss_function=binary_crossentropy,
        verbose=1,
    )

    net1.initialize()

    x_test, y_test = 0, 0
    for epoch in range(num_epochs):
        st = time.clock()
        # make generator
        loo_gen = chb.loo_gen(subj, szr, shuffle=True)
        # get test data (same on every epoch, so not really using it until the
        # last go-round)
        for batch in loo_gen:
            x_test, y_test = batch
            break
        # separate val and train data

        x_train, y_train = 0, 0
        batch_train_errs = []
        for idx, batch in enumerate(loo_gen):
            x_train, y_train = batch
            net1.partial_fit(x_train, y_train)

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

    y_true, y_pred = y_test, net1.predict(x_test)
    print(classfication_report(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))

# Optionally, you could now dump the network weights to a file like this:
# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)
