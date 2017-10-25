# import usefull packages
import numpy as np
import matplotlib.pyplot as plt

def nn_structure(X, Y , L = 2, n_hidden = None):

    '''
    define the structure of Neural network

    Arguments:
    X -- input array of shape(dimension, # of data points)
    Y -- output array of shape(output size, # of data points)
    L -- number of layars eq to 1 + number of hidden layers
    n_hidden -- list containing number of units in each hidden layer
                with length of(L-1)

    Output:
    n_l -- a list of number of units in layers(input, hidden and output layer)
    '''
    n0 = X.shape[0]
    nL = Y.shape[0]

    if L ==1:
        n_l = [n0, nL]
        return structure
    if n_hidden is None :
        # if n_hidden is not given, # of units for all hidden units=
        # 2*dimension of data(number of units in input layer)
        n_hidden =  (L - 1)*[2 * n0]
    else:
        #assert ( type(n_hidden) == list)
        assert (isinstance(n_hidden, list))
        assert ( len(n_hidden) == (L-1) )

    n_l = [n0] + n_hidden + [nL]
    return n_l



def init_params(n_l, seed = 42) :
    ''' initiliza parameters of Neural Network (w, b)

    Arguments:
    structure -- list of number of units for all layers(input, hidden, output)

    Output:
    parameters -- a dictioanry of wi and bi (i=1,...L)
    '''
    L = len(n_l) - 1 # substract input layer
    np.random.seed(seed)
    parameters = {}

    for l in range(1, L+1):
        parameters["W{0}".format(l)] = np.random.randn(n_l[l], n_l[l-1])*.01
        parameters["b{0}".format(l)] = np.zeros((n_l[l], 1))
    return parameters


def sigmoid(Z):
    return 1./(1 + np.exp(-Z))


def forward_propagation(X, parameters, L):
    '''
    forward propogation

    Arguments:
    X -- input array with shape of(dimension, number of data points)
    L -- number of layers eq to 1 + number of hidden layers
    parameters -- a dictioanry of wi and bi (i=1,...L)

    Output :
    Y_hat : predicted values
    cache : a dictioanry of Ai and Zi (i=1,...L) used for trainig NN in backprog
    '''

    AZ = {}
    AZ["A0"] = X
    for l in range(1, L + 1):

        # Z[l] = W[l].A[l-1] + b[l]
        # A[l] = g[l](Z[l])
        AZ["Z{0}".format(l)] = \
        np.dot( parameters["W{0}".format(l)] , AZ["A{0}".format(l-1)]) +\
        parameters["b{0}".format(l)]

        if l != L:
            # for hidden layers, use tanh or relu
            AZ["A{0}".format(l)] = np.tanh(AZ["Z{0}".format(l)])
        else :
            # for output layer use sigmoid(in case of binary classification)
            AZ["A{0}".format(l)] = sigmoid(AZ["Z{0}".format(l)])

    Y_hat = (AZ["A{0}".format(L)])
    #assert (Y_hat.shape == (1, X.shape[1]) )

    return Y_hat, AZ



def compute_j(Y, Y_hat):

    '''
    Arguments :
    Y -- output with shape(output size, number of data points)
    Y_hat __ predicted output with shape(1, number of data points)

    output :  objective function for binary classification( entropy)
    '''
    m = Y.shape[1] # number od data points
    J = (-1./m) * np.sum( (Y*np.log(Y_hat)) + ((1-Y)*np.log(1 - Y_hat)))
    J = np.squeeze(J) # change the type from [[obj]] to obj
    assert (isinstance(J, float))

    return J



def backward_propagation(Y, parameters, AZ, L):

    '''
    Compute dedevatives of parameters for gradient descent

    Arguments:
    Y -- output with shape of (output size, # of data points)
    parameters -- current Wi and bi for i=1,...,L
    AZ -- cache from the forward propogation step

    Output:
    grad -- a dictionary of gradient of Wi and bi for i=1,...,L
    '''

    m = Y.shape[1] # number of data points

    # considering entropy as objective and sigmoid for the output layer:
    # dJ/dZ[L]= dJ/dY_hat * dY_hat/dZ[L] = ... = Y_hat - Y
    dZ = AZ["A{0}".format(L)] - Y

    grads = {}
    for l in range(L,0,-1):
        if l != L:
            dZ = dA * (1 - np.power(AZ["A{0}".format(l)] , 2) )

        grads["dW{0}".format(l)] = (1./m)*np.dot(dZ, AZ["A{0}".format(l-1)].T)
        assert (grads["dW{0}".format(l)].shape == parameters["W{0}".format(l)].shape)

        grads["db{0}".format(l)] = (1./m)*np.sum(dZ, axis =1, keepdims = True)
        assert (grads["db{0}".format(l)].shape == parameters["b{0}".format(l)].shape)

        dA = np.dot(parameters["W{0}".format(l)].T, dZ)

    return grads



def update_parameters(parameters, grads , L, learning_rate = 1.2):
    '''
    given gradients and learning rate, update all parameters in  nn
    Arguments:
    parameters -- a dictionary of wi and bi for i=1,...L
    grads -- a dictionary of gradient of Wi and bi for i=1,...,L
    learning_rate
    L -- number of layers

    output :
    parameters --  a dictionary of updated wi and bi for i=1,...L
    '''

    for l in range(1, L+1):
        parameters["W{0}".format(l)] = parameters["W{0}".format(l)] - \
        learning_rate * grads["dW{0}".format(l)]
        parameters["b{0}".format(l)] = parameters["b{0}".format(l)] - \
        learning_rate * grads["db{0}".format(l)]
    return parameters




def train_nn(X, Y, L = 2, n_hidden = None, n_iteration = 10000,random_state = 42\
,learning_rate = 1.2, print_cost = False):
    ''' train neural network
    Arguments :
    X : input train set with the shape of (# features, #data points)
    Y : output train set with the shape of (# outputs, #data points)
    L : number of layers ( excluding input layer)
    n_hidden : a list of number of units in hidden layers
    n_iteration : number of iteration for gradient descent

    output: parameters for all layers after training
    '''

    # define structyre of the nn by a list of #units for all layers
    n_l = nn_structure(X, Y , L , n_hidden)
    # initial parameters for all layers
    parameters = init_params(n_l, seed = random_state)

    for i in range(n_iteration):
        Y_hat, AZ = forward_propagation(X, parameters, L)
        grads = backward_propagation(Y, parameters, AZ, L)
        parameters = update_parameters(parameters, grads , L, learning_rate)

        if (i%1000 == 0) and print_cost :
            J = compute_j(Y, Y_hat)
            print "Cost after iteration %i: %f" %(i, J)
    return parameters


def predict_class_label(X,parameters, treshold = .5):
    '''
    a function that predict class label for a given input

    arguments:
    X --input
    parameters -- learned wl and bl for l = 1,...,L
    threshold --

    output -- predicted class label
    '''

    L = len(parameters)/2
    Y_hat, AZ = forward_propagation(X, parameters, L)
    prediction = (Y_hat >= treshold).astype(int)
    return prediction
