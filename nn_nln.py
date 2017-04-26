#!/usr/bin/env python
"""
First attempt at Lasagne with my real data.
"""

from __future__ import print_function

import sys
import os
import time
import chb

import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne.nonlinearities import rectify, leaky_rectify, softmax, sigmoid
from lasagne.objectives import binary_crossentropy, binary_accuracy

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
import nolearn.lasagne

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef

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

print('Check using lgus (90% undersampling)')

# Load the dataset
subj = chb.load_dataset(subject, tiger=tiger)
sys.stdout.flush()
batch_size = 10

num_szr = subj.get_num()
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
        output_num_units=1,
        output_nonlinearity=sigmoid,
        # optimization method params
        update=lasagne.updates.rmsprop,
        update_learning_rate=1e-6,
        objective_loss_function=binary_crossentropy,

        on_training_started=nolearn.lasagne.PrintLayerInfo(),
        on_training_finished=nolearn.lasagne.PrintLog(),
        
    )

    net1.initialize()

    for epoch in range(num_epochs):
        loo_gen = chb.lgus(subj, szr, batchsec=60, drop_prob=0.9, shuffle=True)
        x_test, y_test = 0, 0
        for batch in loo_gen:
            x_test, y_test = batch
            break

        for idx, batch in enumerate(loo_gen):
            x_train, y_train = batch
            net1.partial_fit(x_train, y_train)

    y_true, y_pred, y_prob = y_test, net1.predict(x_test), net1.predict_proba(x_test)
    print(classification_report(y_true, y_pred))
    print('Accuracy score:', accuracy_score(y_true, y_pred))
    print('Confusion matrix:\n',confusion_matrix(y_true, y_pred))
    print('ROC-AUC score:',roc_auc_score(y_true, y_pred))
    print('Matthews Coeff:',matthews_corrcoef(y_true, y_pred))
    print('y_true | y_pred | y_prob')
    print(np.array([np.squeeze(y_true), np.squeeze(y_pred), np.squeeze(y_prob)]).T)
# Optionally, you could now dump the network weights to a file like this:
# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)
