import chb
import numpy as np
import theano
from lasagne.layers import (
    InputLayer, Conv2DLayer, DenseLayer, MaxPool2DLayer, DropoutLayer
)
from lasagne.nonlinearities import rectify, leaky_rectify, sigmoid

inshape = (None, 1, 23, 1280)
outshape = 1

def simple(input_var):
    net = {}
    net['data'] = InputLayer(inshape, input_var=input_var)
    net['conv1'] = Conv2DLayer(
        net['data'],
        num_filters=8,
        filter_size=(1, 255),
        stride=(1, 32),
        pad='same',
        nonlinearity=rectify)
    net['conv2'] = Conv2DLayer(
        net['conv1'],
        num_filters=8,
        filter_size=(1, 127),
        pad='same',
        stride=(1, 32),
        nonlinearity=rectify)
    net['pool'] = MaxPool2DLayer(net['conv2'], pool_size=(1, 2))
    net['fcl'] = DenseLayer(net['pool'], num_units=256, nonlinearity=rectify)
    net['out'] = DenseLayer(net['fcl'], num_units=outshape, nonlinearity=sigmoid)
    return net
