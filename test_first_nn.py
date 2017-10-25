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

### checkpoint for above function
# X = np.random.randn(3, 5)
# Y = np.random.randn(1 , 5)
# n_l = nn_structure(X, Y , L = 2, n_hidden = [4, 6])
# print n_l

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

### checkpoint for
# n_l = [3,6,1]
# prameters = init_params(n_l)
# W1 = prameters["W1"]
# W2 = prameters["W2"]
# b1 = prameters["b1"]
# b2 = prameters["b2"]
# print W1
# print b1.shape
# print W2.shape
# print b2.shape
# print prameters["W{0}".format(1)]

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

### check forward propogation
# def forward_propagation_test_case():
#     np.random.seed(1)
#     X_assess = np.random.randn(2, 3)
#     b1 = np.random.randn(4,1)
#     b2 = np.array([[ -1.3]])
#
#     parameters = {'W1': np.array([[-0.00416758, -0.00056267],
#         [-0.02136196,  0.01640271],
#         [-0.01793436, -0.00841747],
#         [ 0.00502881, -0.01245288]]),
#      'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
#      'b1': b1,
#      'b2': b2}
#
#     return X_assess, parameters
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters, 2)
#
# # Note: we use the mean here just to make sure that your output matches ours.
# print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

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

### check compute j
# def compute_cost_test_case():
#     np.random.seed(1)
#     Y_assess = (np.random.randn(1, 3) > 0)
#     parameters = {'W1': np.array([[-0.00416758, -0.00056267],
#         [-0.02136196,  0.01640271],
#         [-0.01793436, -0.00841747],
#         [ 0.00502881, -0.01245288]]),
#      'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
#      'b1': np.array([[ 0.],
#         [ 0.],
#         [ 0.],
#         [ 0.]]),
#      'b2': np.array([[ 0.]])}
#
#     a2 = (np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]))
#
#     return a2, Y_assess, parameters
#
# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_j( Y_assess, A2)))


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

###check backpropogation
# def backward_propagation_test_case():
#     np.random.seed(1)
#     X_assess = np.random.randn(2, 3)
#     Y_assess = (np.random.randn(1, 3) > 0)
#     parameters = {'W1': np.array([[-0.00416758, -0.00056267],
#         [-0.02136196,  0.01640271],
#         [-0.01793436, -0.00841747],
#         [ 0.00502881, -0.01245288]]),
#      'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
#      'b1': np.array([[ 0.],
#         [ 0.],
#         [ 0.],
#         [ 0.]]),
#      'b2': np.array([[ 0.]])}
#
#     cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
#          [-0.05225116,  0.02725659, -0.02646251],
#          [-0.02009721,  0.0036869 ,  0.02883756],
#          [ 0.02152675, -0.01385234,  0.02599885]]),
#   'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
#   'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
#          [-0.05229879,  0.02726335, -0.02646869],
#          [-0.02009991,  0.00368692,  0.02884556],
#          [ 0.02153007, -0.01385322,  0.02600471]]),
#   'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}
#     return parameters, cache, X_assess, Y_assess
#
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
# cache['A0'] = X_assess
# grads = backward_propagation(Y_assess,parameters, cache,2 )
# print grads

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

### check update_parameters
# def update_parameters_test_case():
#     parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
#         [-0.02311792,  0.03137121],
#         [-0.0169217 , -0.01752545],
#         [ 0.00935436, -0.05018221]]),
#  'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
#  'b1': np.array([[ -8.97523455e-07],
#         [  8.15562092e-06],
#         [  6.04810633e-07],
#         [ -2.54560700e-06]]),
#  'b2': np.array([[  9.14954378e-05]])}
#
#     grads = {'dW1': np.array([[ 0.00023322, -0.00205423],
#         [ 0.00082222, -0.00700776],
#         [-0.00031831,  0.0028636 ],
#         [-0.00092857,  0.00809933]]),
#  'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03,
#           -2.55715317e-03]]),
#  'db1': np.array([[  1.05570087e-07],
#         [ -3.81814487e-06],
#         [ -1.90155145e-07],
#         [  5.46467802e-07]]),
#  'db2': np.array([[ -1.08923140e-05]])}
#     return parameters, grads
#
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, L= 2, learning_rate = 1.2)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

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

### check train_nn
# def nn_model_test_case():
#     np.random.seed(1)
#     X_assess = np.random.randn(2, 3)
#     Y_assess = (np.random.randn(1, 3) > 0)
#     return X_assess, Y_assess
# X_assess, Y_assess = nn_model_test_case()
# parameters = train_nn(X_assess, Y_assess, L =2, n_hidden = [4], random_state =2,
#  print_cost=True)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def predict(X,parameters, treshold = .5):
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
