from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, inv
from scipy.linalg import block_diag, cholesky, sqrtm

from topwave.types import RealList, SquareMatrix

__all__ = ["colpa"]

def colpa(H: SquareMatrix) -> tuple[RealList, SquareMatrix]:
    """Diagonalizes a bosonic Hamiltonian.
    

    .. admonition:: Caution!
        :class: caution

        This routine relies on the Hamiltonian to be positive (semi)definite. If the magnetic configuration in the model
        does not minimize the classical energy (e.g. in frustrated systems) that is not the case.

    Parameters
    ----------
    H : SquareMatrix
        Bosonic positive (semi)definite Hamiltonian.

    Returns
    -------
    tuple[RealList, SquareMatrix]
        Eigenvalues w and Eigenvectors v. The column v[:, i] corresponds to the eigenvalue w[i].

    See Also
    --------
    :class:`topwave.spec.Spec`

    References
    ----------
    The function uses the algorithm presented in https://doi.org/10.1016/0378-4371(86)90056-7.

    """
    
    K = cholesky(H)

        
    # build commuation relation matrix and construct commutation-relation preserving auxilary Hamiltonian
    bos = block_diag(np.eye(H.shape[0]//2), -np.eye(H.shape[0]//2))
    L = K @ bos @ K.T.conj()
    
    # diagonalize it
    E, U = eigh(L)
    
    # sort the eigenvalues to decreasing order
    E = E[::-1]
    # NOTE: Check the sorting of the modes. It seems to be consistent with the correlation function like this though.
    U = U[:, :]

    T = inv(K) @ U @ sqrtm(bos * np.diag(E))
    
    return E, T

