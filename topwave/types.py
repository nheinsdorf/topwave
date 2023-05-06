import numpy as np

ComplexVector = np.ndarray[np.complex128] | list[complex]
Complex2x2 = np.ndarray[np.complex128]
IntVector = np.ndarray[np.int64] | list[int]
Matrix = np.ndarray[np.complex128]
Real3x3 = np.ndarray[np.float64]
Vector = np.ndarray[np.float64] | list[float]
VectorList = np.ndarray[np.float64] | list[np.ndarray[np.float64]] | list[list[float]]