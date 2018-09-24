#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """
    #print("params =", params)
   #params = params[:, np.newaxis]
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H)) # shape = 1 x 5
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy)) # shape = 5 x 10
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy)) # shape = 1 x 10

    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data, W1) + b1 # [20 x 10] x [10 x 5] = [20 x 5]
    a1 = sigmoid(z1) # activated hidden layer shape = [20 x 5]
    z2 = np.dot(a1, W2) + b2 # [20 x 5] x [5 x 10] = [20 x 10]
    a2 = softmax(z2)
    #cost = 1.0/data.shape[0] * np.sum(-1 * np.log(np.sum(np.multiply(a2, labels), axis=1)))
    cost = -np.sum(np.log(a2) * labels)
    # the above is condensed version of the following:
    # element-wise multiplication of the softmaxes and the actual labels
    # summed along the 1st axis; applied natural log 'ln'; summed again
    # and lastly, divided by the number of examples
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation with respect to W1, W2, b1, and b2
    d1 = a2 - labels # shape = 20 x 10 # gradb2 yhat
    d2_1 = np.dot(d1, W2.T) # shape = [20 x 10] x [10 x 5] = [20 x 5]
    d2 = np.dot(a1.T, d1) # [5 x 20] x [20 x 10] = [5 x 10] # gradW2
    d3_1 = np.multiply(d2_1, sigmoid_grad(a1)) # shape = [20 x 5], doesn't change #gradb1
    d4 = np.dot(data.T, d3_1) # [10 x 20] x [20 x 5] = [10 x 5] # gradW1
    # d4_1 = np.dot(d3_1, np.transpose(W1)) # [20 x 5] x [5 x 10] = [20 x 10]
    
    gradW1 = d4
    gradW2 = d2
    gradb1 = np.sum(d3_1, axis=0)
    gradb2 = np.sum(d1, axis=0)
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))
    
    #print("cost =", cost)
   #print("cost2 =", cost2)
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1 # random labels

    # preparing random intitial parameters
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
   # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
#    your_sanity_checks()
