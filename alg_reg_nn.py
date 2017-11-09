# import usefull packages
import numpy as np
import matplotlib.pyplot as plt
import math

def nn_structure(X, Y , L , n_hidden):

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
    with len(L + 1)
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



def init_params(n_l, random_state) :
    ''' initiliza parameters of Neural Network (w, b)
    using "He initialization" avoids vanishing/exploding gradients
    and leads to faster convergence

    Arguments:
    n_l -- structure, list of # of units for all layers(input, hidden, output)
    random_state --

    Output:
    parameters -- a dictioanry of wi and bi (i=1,...L)
    '''
    L = len(n_l) - 1 # number of layers, substract input layer
    np.random.seed(random_state)
    parameters = {}

    for l in range(1, L+1):
        # he initilization, prevents exploding/vanishing gradient
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


def forward_propagation(X, parameters, keep_prob, keep_prob_l,
activation_hidden, random_state):
    '''
    forward propogation (it can support dropout regularization as well)

    Arguments:
    X -- input array with shape of(dimension, number of data points)
    parameters -- a dictioanry of wi and bi (i=1,...L)
    keep_prob -- float in range (0,1], probability of keeping neuron
                keep_prob == 1 without dropout
    keep_prob_l -- a list of (True/False) determines which hidden layers dropout
     apply to with the length of L-1
    activation_hidden : activation function for hidden units (Relu, sigmoid, tanh)
    random_state : used for shutting off neurons

    Output :
    Y_hat : predicted values
    cache : two dictioanry of Ai and Zi (i=1,...L) used for trainig NN in backprog
    '''

    L = len(parameters)//2
    A = {}
    A["A0"] = X # add X to activation dictionary
    Z = {}
    np.random.seed(random_state)
    # if  keep_prob_l is None: apply dropout on all hidden layers
    D = {} # will used if keep_prob < 1
    if (keep_prob < 1) and (keep_prob_l is None):
        keep_prob_l = (L-1)*[1]
    keep_prob_l_helper = [0] + keep_prob_l

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

            if keep_prob < 1:
                if keep_prob_l_helper[l]:
                    # Dl the same shape as Al
                    Dl = np.random.rand( A["A{0}".format(l)].shape[0], \
                    A["A{0}".format(l)].shape[1])
                    # make zero with probability of 1 - keep_prob and restore it
                    D["D{0}".format(l)] = Dl < keep_prob
                    # make zero some activations
                    Al = A["A{0}".format(l)] * D["D{0}".format(l)]
                    # scale activation matrix
                    A["A{0}".format(l)] = Al/ keep_prob

        else :
            # for output layer use sigmoid(in case of binary classification)
            # and we do not dropout
            A["A{0}".format(l)] = non_linear(Z["Z{0}".format(l)], "sigmoid")

    # will use in backpropagation
    if keep_prob == 1:
        cache = (A, Z)
    elif keep_prob < 1:
        cache = (A, Z, D)
    # output of nn
    AL = (A["A{0}".format(L)])
    assert (AL.shape == (1, X.shape[1]) )

    return AL, cache


def compute_j(Y, Y_hat):

    '''
    compute cross entropy objective

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

def compute_j_l2(Y, Y_hat, lamb, parameters):

    '''
    compute objective with l2 regularization

    Arguments :
    Y -- output with shape(output size, number of data points)
    Y_hat -- predicted output with shape(1, number of data points)
    lamb -- l2 regularization parameter
    parameters -- dictionary of weights and bias terms

    output :  objective function for binary classification( entropy)
    '''
    L = len(parameters) //2
    m = Y.shape[1] # number od data points
    # cross entropy cost
    J1 = (-1./m) * np.sum( (Y*np.log(Y_hat)) + ((1-Y)*np.log(1 - Y_hat)))
    J1 = np.squeeze(J) # change the type from [[obj]] to obj
    assert (isinstance(J1, float))

    # regularization cost
    J_F = 0
    for l in range(1, L+1 ):
        J_F = np.sum(parameters['W' + str(l)]**2) + J_F
    J2 = (lamb/ (2.*m ) ) * J_F
    assert (isinstance(J_F, float))

    J = J1 + J2
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

def backward_propagation(Y, parameters, cache,lamb, keep_prob, keep_prob_l,
activation_hidden):

    '''
    Compute dedevatives of parameters for gradient descent , supports dropout
    and l2 regularization as well

    Arguments:
    Y -- output with shape of (output size, # of data points)
    parameters -- current Wi and bi for i=1,...,L
    cache  -- tuple of (A,Z) for all layers from the forward propogation step
            and in case keep_prob< 1, (A,Z,D)
    lamb -- l2 regularization parameter
    keep_prob -- float in range (0,1], probability of keeping neuron
                keep_prob == 1 without dropout
    keep_prob_l -- a list of (True/False) determines which hidden layers dropout
     apply to with the length of L-1
    activation_hidden : activation function for hidden units (Relu, sigmoid, tanh)

    Output:
    grad -- a dictionary of gradient of Wi and bi for i=1,...,L
    '''
    # number of layers
    L = len(parameters)//2
    m = Y.shape[1] # number of data points
    A = cache[0]
    Z = cache[1]
    if keep_prob < 1:
        D = cache[2]
    keep_prob_l_here = [0] + keep_prob_l
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

        if lamb == 0 :
            grads["dW{0}".format(l)] = (1./m)*np.dot(dZ, A["A{0}".format(l-1)].T)
        else :
            grads["dW{0}".format(l)] = (1./m)*np.dot(dZ, A["A{0}".format(l-1)].T)\
             + (lamb/m) * parameters['W' + str(l)]

        assert (grads["dW{0}".format(l)].shape == parameters["W{0}".format(l)].shape)

        grads["db{0}".format(l)] = (1./m)*np.sum(dZ, axis =1, keepdims = True)
        assert (grads["db{0}".format(l)].shape == parameters["b{0}".format(l)].shape)

        # derevative of linear part
        dA = np.dot(parameters["W{0}".format(l)].T, dZ)
        if keep_prob < 1:
            if keep_prob_l_here[l-1]: # it is dA(l-1)
                dA = dA * D["D{0}".format(l-1)]
                dA  = dA / keep_prob

    return grads


def update_parameters(parameters, grads , learning_rate ,\
 S_dwdb, V_dwdb, beta1, beta2, epsilon, iter_mimi_batch):
    '''
    update all parameters in  nn, it uses Adam optimization technique

    Arguments:
    parameters -- a dictionary of wi and bi for i=1,...L
    grads -- a dictionary of gradient of Wi and bi for i=1,...,L
    learning_rate
    S_dwdb -- second momentum (RMS prop)
    beta2 -- related parameter to S
    V_dwdb -- first momentum (momentum technique)
    beta1 -- related parameter to V
    epsilon -- small number fo computational stability
    iter_mimi_batch -- total number of mii batch passed so far

    output :
    parameters --  a dictionary of updated wi and bi for i=1,...L
    '''
    L = len(parameters)//2

    for l in range(1, L+1):
        assert(V_dwdb["W{0}".format(l)].shape == grads["dW{0}".format(l)].shape)
        # momentun
        V_dwdb["W{0}".format(l)] = beta1 * V_dwdb["W{0}".format(l)] +
        (1 - beta1)*grads["dW{0}".format(l)]

        V_dwdb["b{0}".format(l)] = beta1 * V_dwdb["b{0}".format(l)] +
        (1 - beta1)*grads["db{0}".format(l)]
        # RMS prob
        S_dwdb["W{0}".format(l)] = beta2 * S_dwdb["W{0}".format(l)] +
        (1 - beta2)* (grads["dW{0}".format(l)]**2)

        S_dwdb["b{0}".format(l)] = beta2 * S_dwdb["b{0}".format(l)] +
        (1 - beta2)* (grads["db{0}".format(l)] **2)

        #Bias correction
        V_dwdb["W{0}".format(l)] = \
        V_dwdb["W{0}".format(l)]/(1 - math.pow(beta1, iter_mimi_batch))

        V_dwdb["b{0}".format(l)] = \
        V_dwdb["b{0}".format(l)]/(1 - math.pow(beta1, iter_mimi_batch))

        S_dwdb["W{0}".format(l)] = \
        S_dwdb["W{0}".format(l)]/(1 - math.pow(beta2, iter_mimi_batch))

        S_dwdb["b{0}".format(l)] = \
        S_dwdb["b{0}".format(l)]/(1 - math.pow(beta2, iter_mimi_batch))

        # Adam updating rule
        parameters["W{0}".format(l)] = parameters["W{0}".format(l)] - \
        learning_rate * np.divide(V_dwdb["W{0}".format(l)], np.sqrt(S_dwdb["W{0}".format(l)]) +epsilon)

        parameters["b{0}".format(l)] = parameters["b{0}".format(l)] - \
        learning_rate * np.divide(V_dwdb["b{0}".format(l)], np.sqrt(S_dwdb["b{0}".format(l)]) +epsilon)

    return parameters




def train_nn(X_original, Y_original, L = 2, n_hidden = None, n_epoch = 1000,\
,learning_rate = 1.2, print_cost = False, activation_hidden = "Relu",
keep_prob =1., keep_prob_l = None, lamb = 0., random_state = 42
batch_size = 2**6, alpha0 = .2, decay_rate = 1,
beta1 = .9, beta2 = .999, epsilon = 1e-8):
    ''' train neural network
    Arguments :
    X : input train set with the shape of (# features, #data points)
    Y : output train set with the shape of (# outputs, #data points)
    L : number of layers ( excluding input layer)
    n_hidden : a list of number of units in hidden layers
    n_iteration : number of iteration for gradient descent
    learning_rate : learning rate for gradient descent
    activation_hidden : activation function for hidden layers, it can be relu,
    tanh or sigmoid
    lamb : regularization coefficient
    keep_prob : another regularization technique( probability of keepimh neurons)
    keep_prob_l : list of layers that dropout applys on with the lenght of ( L-1)

    output: parameters for all layers after training
    '''
    # make a copy of x and y
    X = np.copy(X_original)
    Y = np.copy(Y_original)

    m = X.shape[1] # number of data points

    # shuffle data
    np.random.seed(seed)
    indices = np.random.permutation(m)
    X = X[:, indices]
    Y = Y[:, indices]

    assert (lamb == 0. or keep_prob ==1.) # use just one of regularization technique

    assert( beta1 <1 and beta2<1 and epsilon<1e-6)

    costs = [] # for storing cost at 100 epoch

    T = math.ceil(float(m )/ batch_size)     # number of batches

    # define structyre of the nn by a list of #units for all layers
    n_l = nn_structure(X, Y , L , n_hidden)
    # initial parameters for all layers
    parameters = init_params(n_l, seed = random_state)

    # initialize first and second momentum by 0 for adams optimization algorithm
    # S and V hase shame shape as grads and parameters
    S_dwdb = dict.fromkeys(parameters, 0)
    V_dwdb = dict.fromkeys(parameters, 0)

    # loop for gradient descent converge
    for i in range(n_epoch):
        for t in range(T):

            iter_mimi_batch = T *i + t
            if t != T -1:
                window = range(t*batch_size , (t + 1)*batch_size)
            else :
                window = range(t*batch_size, m)

            Y_hat, cache = forward_propagation(X[:, window],
            parameters, keep_prob, keep_prob_l, activation_hidden, random_state)

            grads = backward_propagation(Y[0, window], parameters, cache,
            lamb, keep_prob, keep_prob_l,activation_hidden)

            learning_rate = (1./(1 + (decay_rate * i)))* alpha0

            parameters = update_parameters(parameters, grads ,learning_rate,
            S_dwdb, V_dwdb, beta1, beta2, epsilon, iter_mimi_batch)

        if (i%100 == 0) and print_cost :
            if lamb == 0.:
                J = compute_j(Y, Y_hat)
            else:
                J = compute_j_l2(Y, Y_hat, lamb, parameters)

            print "Cost after iteration %i: %f" %(i, J)
            costs.append(J)

    if print_cost:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration*100')
        plt.title(' learning_rate =' + str(learning_rate) + '\n $\lambda' + str(lamb))
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
