#%%
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
from lasagne.nonlinearities import rectify, leaky_rectify, sigmoid
from lasagne.objectives import binary_crossentropy

#%%
# Load data for subject
subject = chb.CHBsubj()
subject.load_meta('chb03')
subject.load_data(exthd=False)

#%%
# Load and read training and test set images and labels.
x_train, y_train, x_test, y_test = chb.leaveOneOut(subject, 1, 1000, 100)

# We reserve the last 100 training examples for validation.
x_train, x_val = x_train[:-100], x_train[-100:]
y_train, y_val = y_train[:-100], y_train[-100:]

#%%
data_size = (None, 1, 23, 256)
output_size = 1 # Not sure if this is right! Maybe it should be 2?

#%%
def scratch_net(input_var):

    net = {}
    net['data']  = layers.InputLayer(data_size, input_var=input_var)
    net['conv1'] = layers.Conv2DLayer(net['data'],  num_filters=4, filter_size=(1,7), pad='same',
                                      nonlinearity=rectify)
    net['conv2'] = layers.Conv2DLayer(net['conv1'], num_filters=8, filter_size=(1,15), pad='same',
                                      stride=(1,2), nonlinearity=rectify)
    net['pool']  = layers.MaxPool2DLayer(net['conv2'], pool_size=(1,2))
    net['fcl']   = layers.DenseLayer(net['pool'], num_units=256, nonlinearity=rectify)
    net['out']   = layers.DenseLayer(net['fcl'], num_units=output_size, nonlinearity=sigmoid)

    return net

#%%
def scratch_model(input_var, target_var, net):

    prediction = layers.get_output(net['out'])
    loss = binary_crossentropy(prediction, target_var)
    loss = lasagne.objectives.aggregate(loss)

    params = layers.get_all_params(net['out'], trainable=True)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=1e-5)

    test_prediction = layers.get_output(net['out'], deterministic=True)

    test_loss = binary_crossentropy(test_prediction, target_var)
    test_loss = lasagne.objectives.aggregate(test_loss)
    test_acc  = T.mean(lasagne.objectives.binary_accuracy(test_prediction, target_var),dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn   = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn

#%%
def scratch_train(train_fn, val_fn):
    train_err_list = []
    val_err_list = []
    val_acc_list = []

    print('| epoch \t| train loss\t| val loss\t| val acc\t| time')
    print('='*80)

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
        print('| %d \t\t| %.6f\t| %.6f\t| %.2f%%\t| %.2f s'
              % (epoch+1, epoch_train_err, epoch_val_err, epoch_val_acc * 100, en-st))
        #print('-'*80)
    return train_err_list, val_err_list, val_acc_list

#%%
def scratch_test(val_fn):

    print('Test Results:')
    print('='*80)

    batch_err = []
    batch_acc = []
    for batch in iterate_minibatches(x_test, y_test, batch_size):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        batch_err.append(err)
        batch_acc.append(acc)

    test_err = np.mean(batch_err)
    test_acc = np.mean(batch_acc)

    print('Test loss: %.6f \t| Test accuracy: %.2f' % (test_err, test_acc * 100))
    return test_err, test_acc

#%%
def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

#%%
num_epochs=50
batch_size=10

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
net = scratch_net(input_var)
train_fn, val_fn = scratch_model(input_var, target_var, net)

train_err, val_err, val_acc = scratch_train(train_fn, val_fn)

#%%
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure()
plt.plot(range(num_epochs), train_err, label='Training error')
plt.plot(range(num_epochs), val_err, label='Validation error')
plt.title('ConvNet Training')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
#plt.show()

fig2 = plt.figure()
plt.plot(range(num_epochs), np.asarray(val_acc) * 100)
plt.title('ConvNet Training: Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.show()

#%%
fig, ax1 = plt.subplots()
ax1.plot(range(num_epochs), train_err, label='Training Error')
ax1.plot(range(num_epochs), val_err, label='Validation Error')

ax2 = ax1.twinx()
ax2.plot(range(num_epochs), np.asarray(val_acc) * 100, color='g', label='Validation accuracy')

#%%
test_err, test_acc = scratch_test(val_fn)
