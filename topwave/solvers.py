#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:44:41 2022

@author: niclas
"""
import numpy as np
from numpy.linalg import eig, eigh, inv
from scipy.linalg import block_diag, cholesky, sqrtm

def colpa(H):
    """ Diagonalizes the Hamiltonian using the algorithm by Colpa
    

    Parameters
    ----------
    H : numpy.ndarray
        Bosonic Hamiltonian that is diagonalized.

    Returns
    -------
    Eigenvalues and Eigenvectors.

    """
    
    K = cholesky(H)

        
    # build commuation relation matrix and construct commutation relation
    # preserving auxilary Hamiltonian
    bos = block_diag(np.eye(H.shape[0]//2), -np.eye(H.shape[0]//2))
    L = K @ bos @ K.T.conj()
    
    # diagonalize it
    E, U = eigh(L)
    
    # sort the eigenvalues to decreasing order
    E = E[::-1]
    U = U[:, ::-1]
    
    T = inv(K) @ U @ sqrtm(np.diag(E))
    
    return E, T

