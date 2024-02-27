from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import det

from topwave.types import IntVector, Matrix, RealList, SquareMatrixList
from topwave.util import get_berry_curvature_indices

if TYPE_CHECKING:
    from topwave.spec import Spec


def get_berry_curvature(
        spec: Spec,
        component: str) -> list[RealList]:
    r"""Computes the Berry curvature using

    .. math:: \Omega_k = - \sum_{n ≠ m} \frac{\langle \psi_n \partial_\mu \hat{H} \psi_m \rangle}{(\epsilon_n - \epsilon_m)^2}


    This is so far only for the fermionic berry curvature.

    Examples
    --------
    we compute

    """

    index1, index2 = get_berry_curvature_indices(component)
    partial_derivative_matrix1 = spec.get_tightbinding_derivative(spec.model,
                                                                  spec.kpoints.kpoints,
                                                                  derivative='xyz'[index1])
    partial_derivative_matrix2 = spec.get_tightbinding_derivative(spec.model,
                                                                  spec.kpoints.kpoints,
                                                                  derivative='xyz'[index2])

    num_bands = spec.energies.shape[1]

    M_munu = np.einsum('lij, ljk -> lik',
                   np.einsum('ikl, ikm -> ilm', spec.psi.conj(), partial_derivative_matrix1),
                   spec.psi)
    M_numu = np.einsum('lij, ljk -> lik',
                     np.einsum('ikl, ikm -> ilm', spec.psi.conj(), partial_derivative_matrix2),
                     spec.psi)

    # calculate the square of all the energy differences
    # and add a large number on the diagonal to make the diagonal terms vanish when taking the inverse.
    very_large_number = 1e+28
    index_span = np.arange(num_bands)
    band_index1, band_index2 = np.meshgrid(index_span, index_span, indexing='ij')
    Delta_E = np.square(spec.energies[:, band_index1] - spec.energies[:, band_index2]) \
              + (np.eye(num_bands) * very_large_number)[None, ...]

    return np.real(1j * (M_munu * M_numu.swapaxes(1, 2) - M_numu * M_munu.swapaxes(1, 2)) * np.reciprocal(Delta_E)).sum(axis=2)

def get_bosonic_berry_curvature(spec: Spec,
                                component: str) -> list[RealList]:
    r"""Computes the bosonic Berry curvature.


    Note make a switch for bosonic vs fermionic in ONE function.

    Examples
    --------
    we compute

    """

    index1, index2 = get_berry_curvature_indices(component)
    partial_derivative_matrix1 = spec.get_spinwave_derivative(spec.model,
                                                              spec.kpoints.kpoints,
                                                              derivative='xyz'[index1])
    partial_derivative_matrix2 = spec.get_spinwave_derivative(spec.model,
                                                              spec.kpoints.kpoints,
                                                              derivative='xyz'[index2])

    num_bands = spec.energies.shape[1]

    M_munu = np.einsum('lij, ljk -> lik',
                   np.einsum('ikl, ikm -> ilm', spec.psi.conj(), partial_derivative_matrix1),
                   spec.psi)
    M_numu = np.einsum('lij, ljk -> lik',
                     np.einsum('ikl, ikm -> ilm', spec.psi.conj(), partial_derivative_matrix2),
                     spec.psi)

    # calculate the square of all the energy differences
    # and add a large number on the diagonal to make the diagonal terms vanish when taking the inverse.
    very_large_number = 1e+28
    index_span = np.arange(num_bands)
    band_index1, band_index2 = np.meshgrid(index_span, index_span, indexing='ij')
    Delta_E = np.square(spec.energies[:, band_index1] - spec.energies[:, band_index2]) \
              + (np.eye(num_bands) * very_large_number)[None, ...]

    return np.real(1j * (M_munu * M_numu.swapaxes(1, 2) - M_numu * M_munu.swapaxes(1, 2)) * np.reciprocal(Delta_E)).sum(axis=2)

def get_berry_phase(loop_operator: Matrix) -> float:
    """Computes the Berry phase of a Wilson loop:

    .. math:: \phi = - \sum \operatorname{Im} \ln \det M^{(\Lambda_{i}, \Lambda_{i+1})},


    with

    .. math:: M^{(\Lambda_{i}, \Lambda_{i+1})}_{mn} = \langle u_m^{(\Lambda_{i})} u_n^{(\Lambda_{i+1})} \rangle

    Examples
    --------
    Compute the Berry phase of something.


    """

    return -1 * np.angle(det(loop_operator))


def get_bosonic_wilson_loop(spectrum: Spec, band_indices: IntVector) -> Matrix:
    """Constructs the Wilson loop operator of a bosonic spectrum.

    """

    band_indices = np.arange(spectrum.energies.shape[1]) if band_indices is None else np.array(band_indices,
                                                                                               dtype=np.int64)
    energies, psi = spectrum.energies[:, band_indices], spectrum.psi[:, :, band_indices]
    # check whether start and end k-point are the same and impose closed loop

    # should I check this or give resonsibility to user?
    if np.all(np.isclose(spectrum.kpoints.kpoints[0], spectrum.kpoints.kpoints[-1])):
        psi[0] = psi[-1]
    else:
        # implement the case where they are connected by a reciprocal vector
        # https://github.com/bellomia/PythTB/blob/master/pythtb.py
        # see 'impose_pbc'-method
        psi[0] = psi[-1]
        # pass

    bdg_sign_matrix = np.diag(np.concatenate((np.ones(psi.shape[1] // 2), -np.ones(psi.shape[1] // 2))))
    # construct bra-eigenvectors for k+1
    psi_left = np.roll(np.conj(psi), 1, axis=0)
    psi_right = np.einsum('ij, kin -> kjn', bdg_sign_matrix, psi)

    # compute num_k - 1 overlaps
    loop = np.einsum('knm, knl -> kml', psi_left[1:], psi_right[1:])
    # do the SVD cleanup?
    # take the product of the matrices to compute the wilson loop operator
    return np.linalg.multi_dot(loop)

def get_fermionic_wilson_loop(spectrum: Spec,
                              band_indices: IntVector = None,
                              energy: float = None) -> Matrix:
    """Constructs the Wilson loop operator of a fermionic spectrum.

    For a spectrum at Nk k-points, the inner product of Nk - 1 eigenfunctions for a given selection of
    states is evaluated. The ordering is the same as that of the k-points. The states can be selected by providing a
    list of band indices, or all states below some energy can be selected. In that case the wilson loop operator is
    a product of rectangular matrices.

    .. admonition:: Careful!
        :class: warning

        Typically, **closed** Wilson loops of the **occupied** states are the desired quantities. Make sure the
        last and first k-point of the spectrum are the same, and all selected bands are
        separated in energy (**nondegenerate**).

    Parameters
    ----------
    spectrum: Spec
        The spectrum that contains the eigenfunctions of the model.
    band_indices: IntVector
        List of band indices that are selected to compute the Wilson loop operator. If None, all states are selected.
    energy: float
        If not None, only states below energy are used to compute the Wilson loop. Default is None.

    """

    band_indices = np.arange(spectrum.energies.shape[1]) if band_indices is None else np.array(band_indices, dtype=np.int64)
    energies, psi = spectrum.energies[:, band_indices], spectrum.psi[:, :, band_indices]
    # check whether start and end k-point are the same and impose closed loop

    # should I check this or give resonsibility to user?
    if np.all(np.isclose(spectrum.kpoints.kpoints[0], spectrum.kpoints.kpoints[-1])):
        psi[0] = psi[-1]
    else:
        # implement the case where they are connected by a reciprocal vector
        # https://github.com/bellomia/PythTB/blob/master/pythtb.py
        # see 'impose_pbc'-method
        psi[0] = psi[-1]
        #pass

    # standard
    if energy is None:
        # construct bra-eigenvectors for k+1
        psi_left = np.roll(np.conj(psi), 1, axis=0)

        # compute num_k - 1 overlaps
        loop = np.einsum('knm, knl -> kml', psi_left[1:], psi[1:])

        # do the SVD cleanup?
        # take the product of the matrices to compute the wilson loop operator
        return np.linalg.multi_dot(loop)

    # Ken's method
    psi_left = np.conj(psi[0, :, energies[0] <= energy].T)
    psi_right = psi[1, :, energies[1] <= energy].T
    product = np.einsum('nm, nl -> ml', psi_left, psi_right)
    print(product.shape)
    for k_index, kpoint in enumerate(spectrum.kpoints.kpoints[1:-1]):
        psi_left = np.conj(psi[k_index + 1, :, energies[k_index + 1] <= energy].T)
        psi_right = psi[k_index + 2, :, energies[k_index + 2] <= energy].T
        intermediate = np.einsum('nm, nl -> ml', psi_left, psi_right)
        product = product @ intermediate
    if product.size == 0:
        return [[0]]
    return product

def get_nonabelian_berry_phase(loop_operator: Matrix) -> float:
    """Computes the Berry phase of a Wilson loop:

    .. math:: \phi = - \sum \operatorname{Im} \ln \det M^{(\Lambda_{i}, \Lambda_{i+1})},


    with

    .. math:: M^{(\Lambda_{i}, \Lambda_{i+1})}_{mn} = \langle u_m^{(\Lambda_{i})} u_n^{(\Lambda_{i+1})} \rangle

    Examples
    --------
    Compute the Berry phase of something.


    """

    return -1 * np.angle(np.linalg.eigvals(loop_operator))


