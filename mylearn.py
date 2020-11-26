####
##
## My Machine Learning Library
##
## Sample code for COSC 522
## 
## Hairong Qi, hqi@utk.edu
####
import numpy as np
import util

def mpp(Tr, yTr, Te, cases, P, output=True):
    """
    Maximum Posterior Probability (MPP):
    Supervised parametric learning assuming Gaussian pdf
    with 3 cases of discriminant functions
    
    Return labels of test samples
    """
    # training process - derive the model
    covs, means = {}, {}     # dictionaries
    covsum = None

    classes = np.unique(yTr)   # get unique labels as dictionary items
    classn = len(classes)    # number of classes
    
    for c in classes:
        # filter out samples for the c^th class
        arr = Tr[yTr == c]  
        # calculate statistics
        covs[c] = np.cov(np.transpose(arr))
        means[c] = np.mean(arr, axis = 0)  # mean along the columns
        # accumulate the covariance matrices for Case 1 and Case 2
        if covsum is None:
            covsum = covs[c]
        else:
            covsum += covs[c]
    
    # used by case 2
    covavg = covsum / classn
    # used by case 1
    varavg = np.sum(np.diagonal(covavg)) / classn
            
    # testing process - apply the learned model on test set 
    disc = np.zeros(classn)
    nr, _ = Te.shape
    y = np.zeros(nr)            # to hold labels assigned from the learned model

    for i in range(nr):
        for c in classes:
            if cases == 1:
                edist2 = util.euc2(means[c], Te[i])
                disc[c] = -edist2 / (2 * varavg) + np.log(P[c] + 0.000001)
            elif cases == 2: 
                mdist2 = util.mah2(means[c], Te[i], covavg)
                disc[c] = -mdist2 / 2 + np.log(P[c] + 0.000001)
            elif cases == 3:
                mdist2 = util.mah2(means[c], Te[i], covs[c])
                disc[c] = -mdist2 / 2 - np.log(np.linalg.det(covs[c])) / 2 + np.log(P[c] + 0.000001)
            else:
                print("Can only handle case numbers 1, 2, 3.")
                sys.exit(1)
        y[i] = disc.argmax()
        if i % 100 == 0 and output:
            print(f"Sample {i}: label is {y[i]}")
            
    return y    


def knn(Tr, yTr, Te, k, output=True):
    """
    k-Nearest Neighbor (kNN): Supervised non-parametric learning
    
    Return labels of test samples
    """
    classes = np.unique(yTr)   # get unique labels as dictionary items
    classn = len(classes)      # number of classes
    ntr, _ = Tr.shape
    nte, _ = Te.shape
    
    y = np.zeros(nte)
    knn_count = np.zeros(classn)
    for i in range(nte):
        test = np.tile(Te[i,:], (ntr, 1))       # resembles MATLAB's repmat function
        dist = np.sum((test - Tr) ** 2, axis = 1) # calculate distance
        idist = np.argsort(dist)    # sort the array in the ascending order and return the index
        knn_label = yTr[idist[0:k]]
        for c in range(classn):
            knn_count[c] = np.sum(knn_label == c)
        y[i] = np.argmax(knn_count)
        if i % 100 == 0 and output:
            print(f"Sample {i}: label is {y[i]}")
        
    return y    


def perceptron(Tr, yTr, Te = None):
    """
    Perceptron: Single-layer neural network for two classes
    
    Return the network weight if Te is not provided;
    Return the label from the trained model if Te is provided.
    """

    nr, nc = Tr.shape          # dimension
    w = np.random.rand(nc + 1) # initial weight
    y = np.zeros(nr)           # output from perceptron

    # training process
    finish = 0
    maxiter = 40
    n = 0        # number of iterations
    while not finish and n < maxiter:
        n += 1
        for i in range(nr):            
            y[i] = np.dot(w[:-1], Tr[i,:]) > w[-1]        # obtain the actual output
            w[:-1] = w[:-1] + (yTr[i] - y[i]) * Tr[i,:]  # online update weight
            w[-1] = w[-1] - (yTr[i] - y[i])       # update bias
        if np.dot(y - yTr, y - yTr) == 0:
            finish = 1
        print(f"Iteration {n}: Actual output from perceptron is: {y}, weights are {w}.")
        
    if Te is None:
        return w
    else:                   # the testing process
        ytest = np.matmul(w[:-1], np.transpose(Te)) > w[-1]
        return ytest.astype(int)


def kmeans(Te, k):
    """
    kmeans clustering: Unsupervised learning 

    Return labels of test samples
    """
    nte, nf = Te.shape    
    centers = np.random.rand(k, nf)   # random initialize center locations
#     centers = np.ones((k, nf))
#     for i in range(k):
#         centers[i, :] = i * 256/k * centers[i, :]
    y = np.zeros(nte)            # to hold labels assigned from the learned model
    
    counter = nte            # count the number of samples changed label
    while counter != 0:
        counter = 0
        for i in range(nte): # one epoch is to go through all samples once
            label = np.argmin(np.sum((centers - np.tile(Te[i], (k, 1))) ** 2, axis = 1))
            if label != y[i]:
                counter += 1
                y[i] = label
        # update cluster center at the end of each epoch
        for c in range(k):
            arr = Te[y == c]  
            centers[c] = np.mean(arr, axis = 0)  # mean across all rows
        print(f'Percentage of changes: {counter/nte}')
        
    return y  


def pca(data, err):
    """
    Principal component analysis (pca): based on the error rate given, decide how many 
    eigenvectors to keep for dimensionality reduction purpose
    
    Return the projection matrix
    """
    assert 0 < err < 1
    
    Sigma = np.cov(np.transpose(data))
    d, E = np.linalg.eig(Sigma)  # d: the eigenvalue array, unsorted, E: the eigenvectors
    id_sorted = np.argsort(d)    # sort d in the ascending order and return the index
    sum_d = np.sum(d)     # sum of eigenvalues
    sum_pd = 0            # sum of partial eigenvalues
    error = 0             # error rate
    i = 0
    while error < err:
        sum_pd += d[id_sorted[i]]
        error = sum_pd / sum_d
        i += 1
    
    return E[:, id_sorted[i-1:]]


def fld(Tr, yTr):
    """
    Fisher's linear discriminant (FLD): reduce the dimension of dataset to c-1,
    where c is the number of classes in the training set
    
    Return the projection matrix
    """
    S, means = {}, {}     # dictionaries
    Sw = None
    Sb = None
    means_all = np.mean(Tr, axis = 0)

    classes = np.unique(yTr)   # get unique labels as dictionary items
    classn = len(classes)         # number of classes
    
    for c in classes:
        # filter out samples for the c^th class
        arr = Tr[yTr == c]  
        n, _ = arr.shape
        # calculate statistics
        S[c] = (n - 1) * np.cov(np.transpose(arr))
        means[c] = np.mean(arr, axis = 0)  # mean along the columns
        # accumulate the scatter matrices 
        if Sw is None:
            Sw = S[c]
        else:
            Sw += S[c]
        if Sb is None:
            Sb = n * np.outer(means[c] - means_all, means[c] - means_all)
        else:
            Sb += n * np.outer(means[c] - means_all, means[c] - means_all)
    
    d, E = np.linalg.eig(np.matmul(np.linalg.inv(Sw), Sb))
    id_sorted = np.argsort(d)    # sort d in the ascending order

    return E[:, id_sorted[-classn+1:]]


#Neilson's neural network

# Standard library
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)
        accPerEpoch = []

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))
            accPerEpoch.append(self.evaluate(test_data) / n_test)
        # Graph of accuracy vs. epoch
        plt.plot(list(range(len(accPerEpoch))), accPerEpoch)
        plt.ylabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Convergence Curve')
        plt.show()
            
            
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))