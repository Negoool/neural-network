''' basic naural network , according to course 1,
without regularization,
batch gradient descent
wihtout gradient checking
based on course 1 '''
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



def init_params(n_l, seed = 2) :
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
        # he initilization
        parameters["W{0}".format(l)] = np.random.randn(n_l[l], n_l[l-1])*\
        np.sqrt(2./n_l[l-1])

        parameters["b{0}".format(l)] = np.zeros((n_l[l], 1))
    return parameters


def non_linear(Z, activation):
    ''' nonlinear activation function
    for output layer, for classification, it is sigmoid
    and for hidden units it is relu'''
    if activation == "sigmoid":
        return 1./(1 + np.exp(-Z))
    if activation == "Relu":
        return np.maximum(0 ,Z)
    if activation == "tanh":
        return np.tanh(Z)

def forward_propagation(X, parameters, activation_hidden = "Relu"):
    '''
    forward propogation

    Arguments:
    X -- input array with shape of(dimension, number of data points)
    L -- number of layers eq to 1 + number of hidden layers
    parameters -- a dictioanry of wi and bi (i=1,...L)

    Output :
    Y_hat : predicted values
    cache : two dictioanry of Ai and Zi (i=1,...L) used for trainig NN in backprog
    '''
    L = len(parameters)//2
    A = {}
    A["A0"] = X
    Z = {}
    # for each layer: recieve A[l - 1], give A[l] (repeat L times)
    for l in range(1, L + 1):
        # linear part
        # Z[l] = W[l].A[l-1] + b[l]
        Z["Z{0}".format(l)] = \
        np.dot( parameters["W{0}".format(l)] , A["A{0}".format(l-1)]) +\
        parameters["b{0}".format(l)]
        # nonlinear part
        # A[l] = g[l](Z[l])
        if l != L:
            # for hidden layers, use tanh or relu
            A["A{0}".format(l)] = non_linear(Z["Z{0}".format(l)], activation_hidden)
        else :
            # for output layer use sigmoid(in case of binary classification)
            A["A{0}".format(l)] = non_linear(Z["Z{0}".format(l)], "sigmoid")

    cache = (A, Z)
    # output of nn
    AL = (A["A{0}".format(L)])
    #assert (Y_hat.shape == (1, X.shape[1]) )

    return AL, cache


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


def grad_non_linear(Z, activation):
    ''' return derevative g(Z[l]) = dA/dZ'''

    if activation == "sigmoid":
        A = non_linear(Z, activation)
        return A*( 1 - A)
    if activation == "tanh":
        A = non_linear(Z, activation)
        return 1 - np.power(A, 2)
    if activation == "Relu":
        return (Z > 0).astype(int)



def backward_propagation(Y, parameters, cache, activation_hidden = "Relu"):

    '''
    Compute dedevatives of parameters for gradient descent

    Arguments:
    Y -- output with shape of (output size, # of data points)
    parameters -- current Wi and bi for i=1,...,L
    AZ -- cache from the forward propogation step

    Output:
    grad -- a dictionary of gradient of Wi and bi for i=1,...,L
    '''
    # number of layers
    L = len(parameters)//2
    m = Y.shape[1] # number of data points
    A = cache[0]
    Z = cache[1]
    AL = A["A{0}".format(L)] # output

    # initiate backward procedure
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dZL = dAL * grad_non_linear( Z["Z{0}".format(L)], "sigmoid")
    # considering entropy as objective and sigmoid for the output layer:
    # dJ/dZ[L]= dJ/dY_hat * dY_hat/dZ[L] = ... = Y_hat - Y
    dZ = A["A{0}".format(L)] - Y
    assert np.allclose(dZ, dZL)

    grads = {}
    # for each layer, recieve dA(l), compute dZ(l), dw(l), db(l), dA(l-1)
    dA = dAL

    for l in range(L,0,-1):
        if l == L:
            # derevative of nonlinear part
            dZ = dA * grad_non_linear(Z["Z{0}".format(l)], "sigmoid")
        else:
            dZ = dA * grad_non_linear(Z["Z{0}".format(l)], activation_hidden)

        grads["dW{0}".format(l)] = (1./m)*np.dot(dZ, A["A{0}".format(l-1)].T)
        assert (grads["dW{0}".format(l)].shape == parameters["W{0}".format(l)].shape)

        grads["db{0}".format(l)] = (1./m)*np.sum(dZ, axis =1, keepdims = True)
        assert (grads["db{0}".format(l)].shape == parameters["b{0}".format(l)].shape)

        # derevative of linear part
        dA = np.dot(parameters["W{0}".format(l)].T, dZ)

    return grads



def update_parameters(parameters, grads , learning_rate = 1.2):
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
    L = len(parameters)//2
    for l in range(1, L+1):
        parameters["W{0}".format(l)] = parameters["W{0}".format(l)] - \
        learning_rate * grads["dW{0}".format(l)]
        parameters["b{0}".format(l)] = parameters["b{0}".format(l)] - \
        learning_rate * grads["db{0}".format(l)]
    return parameters




def train_nn(X, Y, L = 2, n_hidden = None, n_iteration = 10000,random_state = 42\
,learning_rate = 1.2, print_cost = False, activation_hidden = "Relu"):
    ''' train neural network
    Arguments :
    X : input train set with the shape of (# features, #data points)
    Y : output train set with the shape of (# outputs, #data points)
    L : number of layers ( excluding input layer)
    n_hidden : a list of number of units in hidden layers
    n_iteration : number of iteration for gradient descent

    output: parameters for all layers after training
    '''
    costs = []
    # define structyre of the nn by a list of #units for all layers
    n_l = nn_structure(X, Y , L , n_hidden)
    # initial parameters for all layers
    parameters = init_params(n_l, seed = random_state)

    # loop for gradient descent converge
    for i in range(n_iteration):
        Y_hat, cache = forward_propagation(X, parameters, activation_hidden)
        grads = backward_propagation(Y, parameters, cache, activation_hidden)
        parameters = update_parameters(parameters, grads ,learning_rate)

        if (i%1000 == 0) and print_cost :
            J = compute_j(Y, Y_hat)
            print "Cost after iteration %i: %f" %(i, J)
            costs.append(J)
    if print_cost:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration*1000')
        plt.title(' learning_rate =' + str(learning_rate))
    return parameters


def predict_class_label(X,parameters, Y = None, treshold = .5):
    '''
    a function that predict class label for a given input

    arguments:
    X --input
    parameters -- learned wl and bl for l = 1,...,L
    threshold --

    output -- predicted class label
    '''

    AL, cache = forward_propagation(X, parameters, "Relu")
    prediction = (AL >= treshold).astype(int)

    if Y is not None:
        print "Accuracy : ", np.mean(Y == prediction)
    return prediction
