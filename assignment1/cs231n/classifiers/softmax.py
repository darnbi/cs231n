from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    # numbers of train dataset 
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        #get score class
        scores = X[i,:].dot(W)
        # softmax(get probabilities)
        probabilities = np.exp(scores)/np.sum(np.exp(scores))
        # get cross entrophy loss
        loss -= np.log(probabilities[y[i]])
        
        gradient_q = probabilities.reshape(1,-1)
        #
        gradient_q[0, y[i]] += -1
        
        dW += X[i,:].reshape(-1,1).dot(gradient_q)
       
    #get mean loss
    loss /= num_train
    #regularization(mathemetically equivalent)
    loss += reg*np.sum(W*W)
    
    #get mean gradient
    dW /= num_train
    #regularization
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #get score class
    scores = X.dot(W)
    #get exponential fo scores
    exp_scores = np.exp(scores)
    #get probatilities of each scores
    probabilities = exp_scores / np.sum(exp_scores, axis =1).reshape(-1,1)
    
    #get loss by cross entropy
    loss -= np.sum(np.log(probabilities[range(num_train),y]))
    
    gradient_q = probabilities
    gradient_q[range(num_train),y] += -1
    dW = X.T.dot(gradient_q)
    
    loss /= num_train
    loss += reg*np.sum(W*W)
    
    dW /= num_train
    dW += 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
