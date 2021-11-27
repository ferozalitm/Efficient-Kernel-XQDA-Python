# EK_XQDA.py
import numpy as np

def EK_XQDA(K, probLabels, galLabels):

    numGals = len(galLabels)
    numProbs = len(probLabels)
    n = numGals
    m = numProbs

    # Eq.(11) in paper
    K_xx = K[0:numGals, 0: numGals]
    K_xz = K[0:numGals, numGals: numGals + numProbs]
    K_zx = K[numGals:numGals + numProbs, 0: numGals]
    K_zz = K[numGals:numGals + numProbs, numGals: numGals + numProbs]

    labels = np.unique(np.vstack((galLabels, probLabels)))
    c = len(labels)

    galW = np.zeros((numGals, 1))
    probW = np.zeros((numProbs, 1))
    ni = 0

    F = np.zeros((numGals, 1))
    G = np.zeros((numProbs, 1))
    H = np.zeros((numGals + numProbs, 2 * c))
    num_galClassSum = np.zeros((c, 1))
    num_probClassSum = np.zeros((c, 1))

    for k in range(c):
        galIndex = (galLabels == labels[k]).nonzero()[0].reshape(-1,1)
        nk = len(galIndex)
        num_galClassSum[k,:] = nk

        probIndex = (probLabels == labels[k]).nonzero()[0].reshape(-1,1)
        mk = len(probIndex)
        num_probClassSum[k,:] = mk

        ni = ni + nk * mk
        galW[galIndex] = np.sqrt(mk)
        probW[probIndex] = np.sqrt(nk)

        G[probIndex] = nk
        F[galIndex] = mk
        H[:, [k]] = np.sum(K[:, galIndex.ravel()], axis=1, keepdims=True)
        H[:, [c + k]] = np.sum(K[:, numGals + probIndex.ravel()], axis=1, keepdims=True)

    H_xx = H[0:n, 0:c]
    H_xz = H[0:n, c : 2*c]
    H_zx = H[n : n+m, 0:c]
    H_zz = H[n : n+m, c : 2*c]

    A = np.vstack((K_xx,K_zx)).dot(np.diag(F.ravel())).dot(np.vstack((K_xx,K_zx)).T)      #Eq.(22) in paper
    B = np.vstack((K_xz,K_zz)).dot(np.diag(G.ravel())).dot(np.vstack((K_xz,K_zz)).T)      #Eq.(25) in paper
    C = np.vstack((H_xx,H_zx)).dot(np.hstack((H_xz.T,H_zz.T)))                            #Eq.(30) in paper
    U = m*np.vstack((K_xx,K_zx)).dot(np.vstack((K_xx,K_zx)).T)                            #Eq.(38) in paper
    V = n*np.vstack((K_xz,K_zz)).dot(np.vstack((K_xz,K_zz)).T)                            #Eq.(39) in paper
    E = np.vstack((K_xx,K_zx)).dot(np.ones((n,m))).dot(np.vstack((K_xz,K_zz)).T)          #Eq.(40) in paper

    # Symmetric matrices correction
    A = (A + A.T)/2
    B = (B + B.T)/2
    U = (U + U.T)/2
    V = (V + V.T)/2

    KexCov = U+V-E-E.T-A-B+C+C.T                    # Eq.(45) in paper
    KinCov = A+B-C-C.T                              # Eq.(31) in paper

    # Symmetric matrices correction
    KexCov = (KexCov + KexCov.T)/2
    KinCov = (KinCov + KinCov.T)/2

    ne = (numGals * numProbs) - ni
    KexCov = KexCov / ne
    KinCov = KinCov / ni

    # Matrix regularization
    I1 = np.eye(KinCov.shape[0])
    KinCov = KinCov + (10**-7) * I1

    # Finding eigen system
    eigVl, eigVtr = np.linalg.eig((np.linalg.inv(KinCov)).dot(KexCov))

    # Seperate real eigenvalues and eigenvectors
    sel = (eigVl == eigVl.real)
    eigVl = eigVl[sel]
    eigVl = eigVl.real
    eigVtr = eigVtr[:, sel]
    eigVtr = eigVtr.real

    # Sort eigenvalues and eigenvectors
    idx_sort = eigVl.argsort()[::-1]
    eigVl = eigVl[idx_sort]

    r = np.sum(eigVl > 1)
    qdaDims = np.maximum(1, r)
    theta = eigVtr[:, idx_sort[0:qdaDims]]

    # Normalizing theta to ensure discriminants are unit norm
    for s in range(qdaDims):
        norm_factor = (theta[:,s].T).dot(K).dot(theta[:,s])
        theta[:,s] = theta[:,s]/np.sqrt(norm_factor)

    gamma = np.linalg.inv((theta.T).dot(KinCov).dot(theta)) - np.linalg.inv((theta.T).dot(KexCov).dot(theta))   #From Theorem (3) in paper
    return theta, gamma

def projectPSD(M):
     # project the matrix M to its cone of PSD
     # INPUT:
     #      M: a squared matrix
     # OUTPUT
     #      M: the PSD matrix

     D, V = np.linalg.eig(M)

     if (np.sum(np.isreal(D)) != len(D)) :
         print("Complex Eigen values")
         raise ValueError

     D[D <= 0] = np.finfo(np.float64).eps
     M = V.dot(np.diag(D)).dot(V.T)
     M = (M+M.T)/2
     return M

def mahDist(M, Xg, Xp):
    # Mahalanobis distance
    # Input:
    #   <M>: the metric kernel
    #   <Xg>: features of the gallery samples. Size: [n, d]
    #   [Xp]: features of the probe samples. Optional. Size: [m, d]
    # Output:
    #   dist: the computed distance matrix between Xg and Xp

    u = np.sum(Xg.dot(M) * Xg, axis=1, keepdims=True)
    v = np.sum(Xp.dot(M) * Xp, axis=1, keepdims=True)
    dist = u + v.T
    dist = dist - 2 * Xg.dot(M).dot(Xp.T)
    return dist