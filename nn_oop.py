''' neural network in a object oriented version compatible with sklearn pipeline'''
# import useful packages
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class nn(BaseEstimator, ClassifierMixin):
    ''' neural network classifier
    tanh activation function for hidden layers and sigmiod in output layer
    and gradient descent as solver

    methods :
    __init__
    fit(X,Y)
    predict(X)
    predict_proba(X)
    score(X,Y)


    variables :
    n_l_ -- a list of number of units in layers(input, hidden and output layer)
            thet defines the structure of a neural network

    '''
    def __init__(self, L = 2, n_hidden = None, n_iteration = 10000,
    random_state = 42, learning_rate = 1.2, print_cost = False):
        '''
        arguments :
        L : number of layers ( excluding input layer)
        n_hidden : a list of number of units in hidden layers
        n_iteration : number of iteration for gradient descent
        random_state : random integer used for initializing parameters
        learning_rate : constant learning rate for batch gradien descent
        print_cost : print cost after each 1000 iteration (boolean)
        '''
        self.L = L
        self.n_hidden = n_hidden
        self.n_iteration = n_iteration
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.print_cost = print_cost


    def _nn_structure(self, X, Y):

        '''
        define the structure of Neural network by producing n_l_ attributes

        Arguments:
        X -- input array of shape(dimension, # of data points)
        Y -- output array of shape(output size, # of data points)

        '''
        n0 = X.shape[0]
        nL = Y.shape[0]

        if self.L ==1:
            self.n_l_ = [n0, nL]

        if self.n_hidden is None :
            # if n_hidden is not given, # of units for all hidden units=
            # 2*dimension of data(number of units in input layer)
            self.n_hidden =  (self.L - 1)*[2 * n0]
        else:
            #assert ( type(n_hidden) == list)
            assert (isinstance(self.n_hidden, list))
            assert ( len(self.n_hidden) == (self.L-1) )

        self.n_l_ = [n0] + self.n_hidden + [nL]



    def _init_params(self) :
        ''' initiliza parameters of Neural Network (w, b) for all layers

        Output:
        parameters -- a dictioanry of initial wi and bi (i=1,...L)
        '''
        np.random.seed(self.random_state)
        parameters = {}

        for l in range(1, self.L+1):
            parameters["W{0}".format(l)] = \
            np.random.randn(self.n_l_[l], self.n_l_[l-1])*.01
            parameters["b{0}".format(l)] = np.zeros((self.n_l_[l], 1))
        return parameters


    def _sigmoid(self, Z):
        return 1./(1 + np.exp(-Z))


    def _forward_propagation(self, X):
        '''
        forward propogation

        Arguments:
        X -- input array with shape of(dimension, number of data points)

        Output :
        Y_hat : predicted values
        cache : a dictioanry of Ai and Zi (i=1,...L) used for trainig NN in backprog
        '''

        AZ = {}
        AZ["A0"] = X
        for l in range(1, self.L + 1):

            # Z[l] = W[l].A[l-1] + b[l]
            # A[l] = g[l](Z[l])
            AZ["Z{0}".format(l)] = \
            np.dot( self.parameters_["W{0}".format(l)] , AZ["A{0}".format(l-1)])\
             + self.parameters_["b{0}".format(l)]

            if l != self.L:
                # for hidden layers, use tanh or relu
                AZ["A{0}".format(l)] = np.tanh(AZ["Z{0}".format(l)])
            else :
                # for output layer use sigmoid(in case of binary classification)
                AZ["A{0}".format(l)] = self._sigmoid(AZ["Z{0}".format(l)])

        Y_hat = (AZ["A{0}".format(self.L)])
        #assert (Y_hat.shape == (1, X.shape[1]) )

        return Y_hat, AZ


    def score(self, Y, Y_hat):
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


    def _backward_propagation(self, Y, AZ):

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
        dZ = AZ["A{0}".format(self.L)] - Y

        grads = {}
        for l in range(self.L,0,-1):
            if l != self.L:
                dZ = dA * (1 - np.power(AZ["A{0}".format(l)] , 2) )

            grads["dW{0}".format(l)] = (1./m)*np.dot(dZ, AZ["A{0}".format(l-1)].T)
            assert (grads["dW{0}".format(l)].shape == self.parameters_["W{0}".format(l)].shape)

            grads["db{0}".format(l)] = (1./m)*np.sum(dZ, axis =1, keepdims = True)
            assert (grads["db{0}".format(l)].shape == self.parameters_["b{0}".format(l)].shape)

            dA = np.dot(self.parameters_["W{0}".format(l)].T, dZ)

        return grads


    def _update_parameters(self, grads):
        '''
        given gradients and learning rate, update all parameters in  nn
        Arguments:
        grads -- a dictionary of gradient of Wi and bi for i=1,...,L

        '''
        for l in range(1, self.L+1):
            self.parameters_["W{0}".format(l)] = self.parameters_["W{0}".format(l)] - \
            self.learning_rate * grads["dW{0}".format(l)]
            self.parameters_["b{0}".format(l)] = self.parameters_["b{0}".format(l)]\
             - self.learning_rate * grads["db{0}".format(l)]



    def fit(self, X, Y):
        ''' train neural network
        '''

        # define structyre of the nn by producing n_l_ attributes
        self._nn_structure(X, Y)
        # initial parameters for all layers
        self.parameters_ = self._init_params()

        for i in range(self.n_iteration):
            Y_hat, AZ = self._forward_propagation(X)
            grads = self._backward_propagation(Y,  AZ)
            self._update_parameters(grads)

            if (i%1000 == 0) and self.print_cost :
                J = self.score(Y, Y_hat)
                print "Cost after iteration %i: %f" %(i, J)
        return self


    def predict(self, X):
        ''' for a given X, predict class label'''
        prob, AZ = self._forward_propagation(X)
        prediction = (prob >= .5).astype(int)
        return prediction

    def predict_proba(self, X):
        ''' for a given X, predict the probability of belonging to esch class'''
        prob, AZ = self._forward_propagation(X)
        return prob


def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = (np.random.randn(1, 3) > 0)
    return X_assess, Y_assess
X_assess, Y_assess = nn_model_test_case()
my_nn = nn(L = 2, n_hidden = [4], n_iteration = 10000,
random_state = 2, learning_rate = 1.2, print_cost = True)
my_nn.fit(X_assess, Y_assess)

print("W1 = " + str(my_nn.parameters_["W1"]))
print("b1 = " + str(my_nn.parameters_["b1"]))
print("W2 = " + str(my_nn.parameters_["W2"]))
print("b2 = " + str(my_nn.parameters_["b2"]))
