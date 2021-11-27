# Kernel functions
import numpy as np

def RBF_kernel(X, Y):

    norm1 = np.square(X).sum(axis=1,keepdims=True)
    norm2 = norm1
    dist1 = np.kron(np.ones((1, X.shape[0])),norm1)
    dist2 = np.kron(np.ones((X.shape[0], 1)), norm2.T)
    dist3 = 2*np.dot(X,X.T)
    dist = dist1+dist2-dist3
    mu = np.sqrt(np.mean(dist)/2)
    K_train = np.exp((-0.5 / (mu**2)) * dist)
    del norm1, norm2, dist1, dist2, dist3, dist

    norm1 = np.square(X).sum(axis=1,keepdims=True)
    norm2 = np.square(Y).sum(axis=1, keepdims=True)
    dist1 = np.kron(np.ones((1, Y.shape[0])),norm1)
    dist2 = np.kron(np.ones((X.shape[0], 1)), norm2.T)
    dist3 = 2*np.dot(X,Y.T)
    dist = dist1+dist2-dist3
    K_test = np.exp((-0.5 / (mu ** 2)) * dist)

    return K_train, K_test, mu


def RBF_kernel2(X, Y, mu):

    norm1 = np.square(X).sum(axis=1,keepdims=True)
    norm2 = np.square(Y).sum(axis=1, keepdims=True)
    dist1 = np.kron(np.ones((1, Y.shape[0])),norm1)
    dist2 = np.kron(np.ones((X.shape[0], 1)), norm2.T)
    dist3 = 2*np.dot(X,Y.T)
    dist = dist1+dist2-dist3
    K_test = np.exp((-0.5 / (mu ** 2)) * dist)

    return K_test
