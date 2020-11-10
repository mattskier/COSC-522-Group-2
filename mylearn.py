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

def mpp(Tr, yTr, Te, cases, P):
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
        if i % 100 == 0:
            print(f"Sample {i}: label is {y[i]}")
            
    return y    


def knn(Tr, yTr, Te, k):
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
        if i % 100 == 0:
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


