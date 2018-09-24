#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad
#%%
def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    
    ### YOUR CODE HERE
    L2 = np.sqrt((x * x).sum(axis=1))
    x = x / L2.reshape(x.shape[0], 1)
    #raise NotImplementedError
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print "All good"

#%%
def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component) - one-hot label
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens [V x d]
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """
    predicted = predicted[:, np.newaxis] # d x 1
    #print(predicted, "pred")
    
    #print(outputVectors, 'outputvec')
    #print(predicted, 'pred')
    
    # [V x d] x [d x 1] = [V x 1]
    yhat = softmax(np.dot(outputVectors, predicted).T).T
    ytarget = np.zeros(yhat.shape) # V x 1
    #print(yhat, 'yhat')
    
    #print(np.dot(outputVectors, predicted).T, 'dot transpose')
    #print(softmax(np.dot(outputVectors, predicted).T), 'soft transpose')
    #print(softmax(np.dot(outputVectors, predicted).T).T, 'soft')
    #print(np.dot(outputVectors, predicted), 'dotprod')
    #print(softmax(np.dot(outputVectors, predicted)), 'softmax of dotprod')
    
    # 
    ytarget[target] = 1
    #print(ytarget, 'ytarg')

    ### YOUR CODE HERE
    cost = -1 * np.sum(np.log(yhat)) # [V x d] x [d x 1] = [V x 1]
    gradPred = np.dot(outputVectors.T, (yhat - ytarget)) # [d x V] x [V x 1] = [d x 1]
    grad = np.dot(predicted, ((yhat - ytarget).T)).T # ([d x 1] x [1 x V]).T =
                                                            # [V x d]
    #print(cost, gradPred, grad)
    ### END YOUR CODE

    return cost, gradPred, grad
#%%

def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """
    """ These are the indexes, not the actual word vectors"""
    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices
#%%

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
   # print(indices, 'indices')

    ### YOUR CODE HERE
    # prepare the negative output word vectors
    negOutputVectors = outputVectors[indices[1]]
    negOutputVectors = negOutputVectors[np.newaxis, :]
    
    #print(K, 'K')
    #print(outputVectors, 'output')
    #print(np.shape(outputVectors))
   # print(len(outputVectors))
    
    
    
    for k in xrange(K-1):
        temp = outputVectors[indices[1+k]]
        temp = temp[np.newaxis, :]
        negOutputVectors = np.concatenate((negOutputVectors, temp))

    #print(negOutputVectors, 'negOutput')    
    
    negOutputVectors = -1 * negOutputVectors
        
    negAndPosOutputVectors = np.concatenate((outputVectors[indices[0]][np.newaxis, :], 
                                             negOutputVectors))
    #print(negOutputVectors, 'negOutput')
    #print(negAndPosOutputVectors, 'negAndPos')
    
    
    # first 'predicted' in the stack is prepared above, then concatenated
    # for the negOutputVectors below
    predStack = predicted.T[np.newaxis, :]
    #predicted = predicted[:, np.newaxis]
    
    for j in xrange(K):
        predStack = np.concatenate((predStack, predicted.T[np.newaxis, :]))
    
    predicted = predicted[:, np.newaxis]
        
    # [1 x d] x [d x 1] = [1 x 1]; [k x d] x [d x 1] = [5 x 1] - summed = [1 x 1]
    cost = (-1 * np.log(sigmoid(np.dot(outputVectors[target, :], predicted)))) - (np.sum(
            np.log(sigmoid(np.dot(negOutputVectors, predicted)))))
    # [1 x d] x [d x 1] = [1 x 1] x [1 x d] = [1 x d] - [1 x d]
    
    gradPred = (((sigmoid(np.dot(outputVectors[target, :], predicted))) - 
                1) * outputVectors[target, :]) - (np.sum(sigmoid(np.dot(
                        negOutputVectors, predicted) - 1)) * (-1 * negOutputVectors))
    
    # gradient with respect to the output word vector
    # gradUo = np.dot((sigmoid(np.dot(outputVectors[target, :], predicted)) - 1), predicted) 
    # [K x d] x [d x 1] = [K x 1] x [1 x d] = [K x d]
    # element-wise multiplication 
    # For neg sample
    negOnesPls = -1 * np.ones((K+1, 1))
    negOnesPls[0] = 1
    
    #print(negAndPosOutputVectors, 'negandPos')
    #print(negOnesPls, 'negones')
    #print(predicted.shape)
    #print(predicted, 'predicted')
    #print(predStack, 'predstack')
    
    grad = np.multiply((np.multiply((sigmoid(np.dot(
            negAndPosOutputVectors, predicted)) - 1), predStack)), negOnesPls)
    ### END YOUR CODE

    return cost, gradPred, grad

#%%
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    
    #print(gradIn, 'gradin') # [V x d]
    #print(gradOut, 'gradout') # [V x d]
    
    # center word vector - [1 x d]
    centerIndex = tokens[currentWord]
    centerWord = inputVectors[centerIndex]
    
    #print(centerIndex)
    
    # find indices of the context words
    #indices = [token[contextWords[0]]]
    indices = [None] * len(contextWords)
    
    for i in xrange(len(contextWords)):
        indices[i] = tokens[contextWords[i]]
    
    # context word vectors - example of 1 - [1 x d]
    #cWord1 = outputVectors[tokens[contextWords[0]]]
    #cWord2 = outputVectors[tokens[contextWords[1]]]
    #cWord3 = outputVectors[tokens[contextWords[2]]]
    #cWord4 = outputVectors[tokens[contextWords[3]]]
    
    # token[contextWord] = target
    
    #cost1, gradIn1, gradOut1 = word2vecCostAndGradient(centerword, indices[0], 
    #                                outputVectors, dataset)
    #print(gradIn, 'gradIn')
    #print(gradIn[centerIndex], 'gradIn center idx')
    
    for index in indices:
        costT, gradInT, gradOutT = word2vecCostAndGradient(centerWord, index, 
                                                           outputVectors, dataset)
        cost = cost + costT
        #print(gradIn, 'gradIn')
        #print(gradIn[centerIndex], 'gradIn[centerindex]')
       # print(gradInT.T.shape, 'gradInT.T')
       # print(gradInT.T, 'gradinT.T')
       #print(np.squeeze(gradInT).shape, 'gradInT shape')
       # print((gradInT.T).shape, 'gradInT shape')
        gradIn[centerIndex] = gradIn[centerIndex] + gradInT.T # grad wrt to centerword
        gradOut = gradOut + gradOutT # grad wrt output words
        
    
    # and so on until all context word costs are calculated

    # use the function above - word2vecCostandGradient
    # cost of skipgram = sum of softmax-CE of each surrounding word
    ### YOUR CODE HERE
    
    
    #cost = 
    #gradIn = 
    #gradOut = 
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()