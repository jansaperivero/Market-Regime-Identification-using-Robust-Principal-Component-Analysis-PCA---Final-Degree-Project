import numpy as np
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
import matplotlib.pyplot as plt

def stage1(X):
    
    n, p = X.shape

    mu_0 = np.mean(X, axis=0)
    X_centered = X-mu_0

    U, D_diag, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    #rank r0
    tolerance = np.finfo(D_diag.dtype).eps * max(n, p) * D_diag[0]
    r0 = np.sum(D_diag > tolerance)

    #Keep only valuable part of matrix
    U_r0 = U[:, :r0]
    D_diag_r0 = D_diag[:r0]
    Vt_r0 = Vt[:r0, :]
    
    #New data matrix
    Z = np.dot(U_r0, np.diag(D_diag_r0))

    return Z, mu_0, Vt_r0, r0

def stage2(Z, r0, k, alpha=0.75):
    
    n, p_r0 = Z.shape

    #Determine size of h the subset
    k_max=10
    h = max(int(alpha * n), (n+k_max+1)//2)

    #Find set H0 of "less outliers" using MCD
    mcd = MinCovDet(support_fraction=h/n).fit(Z)
    